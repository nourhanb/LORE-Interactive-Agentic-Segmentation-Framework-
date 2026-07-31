from torch.utils.data import Dataset, DataLoader
import torchio as tio
import pickle
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator

from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    SpatialPadd,
    DivisiblePadd,
)
from monai.transforms.transform import Transform

import cc3d
import math


# ---------------------------------------------------------------------------
# Custom MONAI transform: per-channel percentile scaling (used for AutoPET)
# ---------------------------------------------------------------------------

class ChannelWisePercentileScale(Transform):
    """Apply percentile scaling separately to each channel of a multi-channel image."""
    def __init__(self, keys, lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, debug=False):
        self.keys = keys if isinstance(keys, list) else [keys]
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.debug = debug
        self._debug_count = 0

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key]
                scaled_channels = []

                if self.debug and self._debug_count < 3:
                    print(f"\n[ChannelWisePercentileScale] Processing {img.shape[0]} channels, shape: {img.shape}")

                for c in range(img.shape[0]):
                    channel_data = img[c]
                    channel_np = channel_data.detach().cpu().numpy()
                    p_lower = np.percentile(channel_np, self.lower)
                    p_upper = np.percentile(channel_np, self.upper)

                    if self.debug and self._debug_count < 3:
                        name = "CT" if c == 0 else "PET" if c == 1 else f"Channel_{c}"
                        print(f"  [{name}] min={channel_np.min():.2f}, max={channel_np.max():.2f}, "
                              f"p_lower={p_lower:.2f}, p_upper={p_upper:.2f}")

                    if p_upper > p_lower:
                        scaled = (channel_data - p_lower) / (p_upper - p_lower) * (self.b_max - self.b_min) + self.b_min
                    else:
                        scaled = torch.zeros_like(channel_data) + self.b_min

                    if self.clip:
                        scaled = torch.clamp(scaled, self.b_min, self.b_max)
                    scaled_channels.append(scaled)

                if self.debug and self._debug_count < 3:
                    self._debug_count += 1

                d[key] = torch.stack(scaled_channels, dim=0)
        return d


# ---------------------------------------------------------------------------
# Dataset_promise — CT-based datasets (colon, pancreas, lits, kits) + AutoPET
# ---------------------------------------------------------------------------

class Dataset_promise(Dataset):
    """
    Unified dataset supporting:
      - colon / pancreas / lits / kits  (CT-based, fixed intensity scaling)
      - autopet                          (PET only, per-channel percentile scaling)
      - brats                            (MRI T1ce + whole-tumor label, percentile scaling)

    HECKTOR uses the separate Dataset_hecktor class below.
    """
    def __init__(self, data, data_dir, split='train', image_size=128, transform=None, pcc=False, args=None):
        self.args = args
        self.data = data.lower()
        self.paths = data_dir

        self.image_size = (image_size, image_size, image_size)
        self.transform = transform
        self.threshold = 0
        self.split = split
        self.pcc = pcc

        self._set_file_paths(self.paths, split)
        self._set_dataset_stat()
        self.monai_transforms = self._get_transforms(split=split)

        self.cc = 1

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        if self.data in ('autopet', 'autopet_lung'):
            return self._getitem_autopet(index)
        if self.data == 'brats':
            return self._getitem_brats(index)
        return self._getitem_generic(index)

    # -----------------------------------------------------------------------
    # AutoPET path
    # -----------------------------------------------------------------------
    def _getitem_autopet(self, index):
        sitk_pet   = sitk.ReadImage(self.pet_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_pet.GetOrigin()    != sitk_label.GetOrigin():    sitk_pet.SetOrigin(sitk_label.GetOrigin())
        if sitk_pet.GetDirection() != sitk_label.GetDirection(): sitk_pet.SetDirection(sitk_label.GetDirection())
        if sitk_pet.GetSpacing()   != sitk_label.GetSpacing():   sitk_label.SetSpacing(sitk_pet.GetSpacing())

        pet_tio   = tio.ScalarImage.from_sitk(sitk_pet)
        label_tio = tio.LabelMap.from_sitk(sitk_label)

        AUTOPET_SPACING = (2.03642011, 2.03642011, 3.0)
        temp_subject = tio.Subject(pet=pet_tio, label=label_tio)
        temp_subject = tio.Resample(target=AUTOPET_SPACING)(temp_subject)
        temp_subject = tio.ToCanonical()(temp_subject)

        pet_tio   = temp_subject.pet
        label_tio = temp_subject.label
        pet_tensor = pet_tio.data  # [1, D, H, W]

        if not hasattr(self, '_debug_autopet_count'):
            self._debug_autopet_count = 0
        if self._debug_autopet_count < 3:
            pet_np = pet_tensor.squeeze().cpu().numpy()
            print(f"\n[AutoPET Debug] Sample {self._debug_autopet_count}: "
                  f"{os.path.basename(self.pet_paths[index])}")
            print(f"  [PET] shape={pet_tensor.shape}, min={pet_np.min():.2f}, "
                  f"max={pet_np.max():.2f}, mean={pet_np.mean():.2f}")
            self._debug_autopet_count += 1

        image_tensor = pet_tensor  # [1, D, H, W]
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor, affine=pet_tio.affine),
            label=label_tio,
        )
        subject_save = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor, affine=pet_tio.affine),
            label=label_tio,
        )

        if self.transform:
            try:
                subject      = self.transform(subject)
                subject_save = self.transform(subject_save)
            except Exception:
                print(self.pet_paths[index])

        if self.pcc:
            subject = self._pcc(subject)

        if subject.label.data.sum() <= self.threshold:
            print(self.pet_paths[index], 'label volume too small')
            if self.split == 'train':
                return self.__getitem__(np.random.randint(self.__len__()))
            else:
                crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
                subject      = crop_transform(subject)
                subject_save = crop_transform(subject_save)
                trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
                return trans_dict["image"].float(), subject.label.data.clone().detach().float(), \
                       self.pet_paths[index], subject_save

        if self.split == "train":
            trans_dict = self.monai_transforms({
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach()
            })[0]
            return trans_dict["image"].float(), trans_dict["label"].float(), self.pet_paths[index]
        else:
            crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
            subject      = crop_transform(subject)
            subject_save = crop_transform(subject_save)
            trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
            return trans_dict["image"].float(), subject.label.data.clone().detach().float(), \
                   self.pet_paths[index], subject_save

    # -----------------------------------------------------------------------
    # BraTS (MICCAI): T1ce MRI + binary whole-tumor mask (label > 0)
    # -----------------------------------------------------------------------
    def _getitem_brats(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin()    != sitk_label.GetOrigin():    sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection(): sitk_image.SetDirection(sitk_label.GetDirection())
        if sitk_image.GetSpacing()   != sitk_label.GetSpacing():   sitk_label.SetSpacing(sitk_image.GetSpacing())

        lab_np = sitk.GetArrayFromImage(sitk_label)
        wt = (lab_np > 0).astype(np.uint8)
        sitk_wt = sitk.GetImageFromArray(wt)
        sitk_wt.CopyInformation(sitk_label)

        img_tio = tio.ScalarImage.from_sitk(sitk_image)
        label_tio = tio.LabelMap.from_sitk(sitk_wt)

        temp_subject = tio.Subject(image=img_tio, label=label_tio)
        temp_subject = tio.ToCanonical()(temp_subject)

        img_tio = temp_subject.image
        label_tio = temp_subject.label
        image_tensor = img_tio.data

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor, affine=img_tio.affine),
            label=label_tio,
        )
        subject_save = tio.Subject(
            image=tio.ScalarImage(tensor=image_tensor, affine=img_tio.affine),
            label=label_tio,
        )

        if self.transform:
            try:
                subject      = self.transform(subject)
                subject_save = self.transform(subject_save)
            except Exception:
                print(self.image_paths[index])

        if self.pcc:
            subject = self._pcc(subject)

        if subject.label.data.sum() <= self.threshold:
            print(self.image_paths[index], 'label volume too small')
            if self.split == 'train':
                return self.__getitem__(np.random.randint(self.__len__()))
            else:
                crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
                subject      = crop_transform(subject)
                subject_save = crop_transform(subject_save)
                trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
                return trans_dict["image"].float(), subject.label.data.clone().detach().float(), \
                       self.image_paths[index], subject_save

        if self.split == "train":
            trans_dict = self.monai_transforms({
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach()
            })[0]
            return trans_dict["image"].float(), trans_dict["label"].float(), self.image_paths[index]

        crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
        subject      = crop_transform(subject)
        subject_save = crop_transform(subject_save)
        trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
        return trans_dict["image"].float(), subject.label.data.clone().detach().float(), \
               self.image_paths[index], subject_save

    # -----------------------------------------------------------------------
    # Generic CT-based path: colon / pancreas / lits / kits
    # -----------------------------------------------------------------------
    def _getitem_generic(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin()    != sitk_label.GetOrigin():    sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection(): sitk_image.SetDirection(sitk_label.GetDirection())
        if sitk_image.GetSpacing()   != sitk_label.GetSpacing():   sitk_label.SetSpacing(sitk_image.GetSpacing())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )
        subject_save = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        if self.data == 'lits':
            b = subject.label.data
            a = tio.CropOrPad._bbox_mask(b[0].cpu().numpy())
            w = int(max(a[1][0] - a[0][0] + 20, 128))
            h = int(max(a[1][1] - a[0][1] + 20, 128))
            d = int(max(a[1][2] - a[0][2] + 20, 128))
            crop_transform = tio.CropOrPad(mask_name='label', target_shape=(w, h, d))
            subject      = crop_transform(subject)
            subject_save = crop_transform(subject_save)
            if self.split == 'train':
                cur = subject.image.data.shape[1:]
                final_shape = tuple(max(s, 128) for s in cur)
                if cur != final_shape:
                    pad = tio.CropOrPad(target_shape=final_shape)
                    subject      = pad(subject)
                    subject_save = pad(subject_save)

        if self.target_label != 0:
            subject      = self._binary_label(subject)
            subject_save = self._binary_label(subject_save)

        if self.transform:
            try:
                subject      = self.transform(subject)
                subject_save = self.transform(subject_save)
            except Exception:
                print(self.image_paths[index])

        if self.pcc:
            subject = self._pcc(subject)

        if subject.label.data.sum() <= self.threshold:
            print(self.image_paths[index], 'label volume too small')
            if self.split == 'train':
                return self.__getitem__(np.random.randint(self.__len__()))
            else:
                # Dict format for sliding-window paths (LiTS + --full_volume).
                if self.data == 'lits' or getattr(self.args, 'full_volume', False):
                    subject = self._pad_min_patch(subject)
                    subject_save = self._pad_min_patch(subject_save)
                    return self._to_lits_dict(subject, subject_save), self.image_paths[index], \
                           self._to_lits_dict(subject_save, subject_save)
                return (subject.image.data.clone().detach().float(),
                        subject.label.data.clone().detach().float(),
                        self.image_paths[index], subject_save)

        # Pad to minimum 128³ before training crops
        if self.split == 'train':
            cur = subject.image.data.shape[1:]
            final_shape = tuple(max(s, 128) for s in cur)
            if cur != final_shape:
                pad = tio.CropOrPad(target_shape=final_shape)
                subject      = pad(subject)
                subject_save = pad(subject_save)

        if self.split == "train":
            trans_dict = self.monai_transforms({
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach()
            })[0]
            return trans_dict["image"].float(), trans_dict["label"].float(), self.image_paths[index]

        else:
            if self.data == 'lits':
                trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
                subject.image.data = trans_dict["image"]
                return self._to_lits_dict(subject, subject_save), self.image_paths[index], \
                       self._to_lits_dict(subject_save, subject_save)

            # Full-FOV mode: no tight 128³ tumor crop. Return a (possibly
            # context-expanded) volume for sliding-window val/test so figures
            # and metrics see real anatomical context.
            if getattr(self.args, 'full_volume', False):
                if self.data == 'kits':
                    # KiTS volumes are huge (often 512² × 600); keep surrounding
                    # anatomy but expand the label bbox instead of the whole CT.
                    subject = self._context_expand(subject)
                    subject_save = self._context_expand(subject_save)
                # Ensure every axis is at least 128 so GridSampler(128) never fails
                # on thin CTs (e.g. pancreas depth < 128 after 1 mm resample).
                subject = self._pad_min_patch(subject)
                subject_save = self._pad_min_patch(subject_save)
                # colon / pancreas: keep the (padded) full volume
                trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
                subject.image.data = trans_dict["image"]
                return self._to_lits_dict(subject, subject_save), self.image_paths[index], \
                       self._to_lits_dict(subject_save, subject_save)

            if self.data == 'kits':
                subject = self._separate_crop(subject)
                subject = self._ensure_spatial_size(subject)

            crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
            subject      = crop_transform(subject)
            subject_save = crop_transform(subject_save)

            image_input = subject.image.data.clone().detach().float()
            trans_dict  = self.monai_transforms({"image": image_input})
            img_aug     = trans_dict["image"].float()

            img_aug      = self._resize_if_needed(img_aug,                             mode='trilinear')
            label_output = self._resize_if_needed(subject.label.data.clone().detach().float(), mode='nearest')

            return img_aug, label_output, self.image_paths[index], subject_save

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _to_lits_dict(self, subject, subject_save):
        """Return the dict format expected by the sliding-window val/test path."""
        def _clean(t):
            t = t.clone().detach().float()
            # TorchIO images are [C, D, H, W]. Drop accidental leading batch dims.
            while t.dim() > 4:
                t = t.squeeze(0)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            return t

        def _affine(a):
            if isinstance(a, torch.Tensor):
                a = a.clone().detach()
                while a.dim() > 2:
                    a = a.squeeze(0)
                return a
            return a.copy() if hasattr(a, 'copy') else a

        return {
            'image': {'data': [_clean(subject.image.data)],
                      'affine': [_affine(subject.image.affine)]},
            'label': {'data': [_clean(subject.label.data)],
                      'affine': [_affine(subject.label.affine)]},
        }

    def _pad_min_patch(self, subject):
        """Pad spatial axes shorter than image_size so sliding-window 128³ works."""
        cur = tuple(int(s) for s in subject.image.data.shape[1:])
        target = tuple(max(s, self.image_size[i]) for i, s in enumerate(cur))
        if cur != target:
            subject = tio.CropOrPad(target_shape=target)(subject)
        return subject

    def _ensure_spatial_size(self, subject):
        """Resize image / label tensors to self.image_size if needed (kits multi-channel case)."""
        def _resize_tensor(t, mode):
            if tuple(t.shape[-3:]) == self.image_size:
                return t
            return torch.nn.functional.interpolate(
                t.unsqueeze(0), size=self.image_size, mode=mode,
                align_corners=False if mode == 'trilinear' else None
            ).squeeze(0)
        subject.image.data = _resize_tensor(subject.image.data.float(), 'trilinear')
        subject.label.data = _resize_tensor(subject.label.data.float(), 'nearest')
        return subject

    def _context_expand(self, subject):
        """
        Expand the label bounding box by --context_margin voxels (default 96)
        so surrounding organs remain visible, without loading an entire KiTS CT.
        Result is at least image_size in every dimension.
        """
        margin = int(getattr(self.args, 'context_margin', 96) or 96)
        label = subject.label.data  # [1, D, H, W]
        coords = torch.nonzero(label[0] > 0, as_tuple=False)
        if coords.numel() == 0:
            # fallback: pad/crop to image_size around volume centre
            return tio.CropOrPad(target_shape=self.image_size)(subject)

        d0, h0, w0 = coords.min(dim=0).values.tolist()
        d1, h1, w1 = coords.max(dim=0).values.tolist()
        D, H, W = label.shape[1:]
        d0 = max(0, d0 - margin); h0 = max(0, h0 - margin); w0 = max(0, w0 - margin)
        d1 = min(D - 1, d1 + margin); h1 = min(H - 1, h1 + margin); w1 = min(W - 1, w1 + margin)
        td = max(d1 - d0 + 1, self.image_size[0])
        th = max(h1 - h0 + 1, self.image_size[1])
        tw = max(w1 - w0 + 1, self.image_size[2])
        # TorchIO CropOrPad centers on the mask — expand the mask temporarily so
        # the crop includes the requested margin around the tumor.
        return tio.CropOrPad(mask_name='label', target_shape=(td, th, tw))(subject)

    def _resize_if_needed(self, t, mode):
        expected = tuple(self.image_size)
        if tuple(t.shape[-3:]) == expected:
            return t
        return torch.nn.functional.interpolate(
            t.unsqueeze(0), size=expected, mode=mode,
            align_corners=False if mode == 'trilinear' else None
        ).squeeze(0)

    def _separate_crop(self, subject):
        label = subject.label.data
        _, N  = cc3d.connected_components(label[0].cpu().numpy(), return_N=True)
        crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)

        if N > 1:
            label_1 = torch.zeros_like(label, dtype=torch.float32)
            label_2 = torch.zeros_like(label, dtype=torch.float32)
            mid_cut  = math.ceil(label.size(1) / 2)
            label_1[0, 0:mid_cut, :]  = label[0, 0:mid_cut, :].float()
            label_2[0, mid_cut:-1, :] = label[0, mid_cut:-1, :].float()

            image_f = subject.image.data.float()
            s1 = tio.Subject(image=tio.ScalarImage(tensor=image_f),  label=tio.LabelMap(tensor=label_1))
            s2 = tio.Subject(image=tio.ScalarImage(tensor=image_f),  label=tio.LabelMap(tensor=label_2))
            s1, s2 = crop_transform(s1), crop_transform(s2)

            if torch.unique(s2.label.data).size(0) == 1:
                subject.image.data, subject.label.data = s1.image.data, s1.label.data
            elif torch.unique(s1.label.data).size(0) == 1:
                subject.image.data, subject.label.data = s2.image.data, s2.label.data
            else:
                subject.image.data = torch.cat([s1.image.data, s2.image.data], dim=0)
                subject.label.data = torch.cat([s1.label.data, s2.label.data], dim=0)
        else:
            subject = crop_transform(subject)
            subject.image.data = subject.image.data.float()
            subject.label.data = subject.label.data.float()

        return subject

    def _set_file_paths(self, data_dir, split):
        split_fname = "split_lung.pkl" if self.data == 'autopet_lung' else "split.pkl"
        split_file = os.path.join(data_dir, split_fname)
        with open(split_file, "rb") as f:
            d = pickle.load(f)[0][split]

        if self.data in ('autopet', 'autopet_lung'):
            self.pet_paths   = []
            self.label_paths = []
            for key in d.keys():
                image_path = d[key][0].strip("/")
                label_path = d[key][1].strip("/")
                if os.path.basename(image_path).startswith('psma_'):
                    continue
                pet_path = image_path.replace("_0000.nii.gz", "_0001.nii.gz")
                pet_full   = os.path.join(data_dir, pet_path)
                label_full = os.path.join(data_dir, label_path)
                if not os.path.exists(pet_full):
                    raise FileNotFoundError(f"PET file not found: {pet_full}")
                if not os.path.exists(label_full):
                    raise FileNotFoundError(f"Label file not found: {label_full}")
                self.pet_paths.append(pet_full)
                self.label_paths.append(label_full)
            if len(self.pet_paths) > 500:
                self.pet_paths   = self.pet_paths[:500]
                self.label_paths = self.label_paths[:500]
                print(f"Limited {self.data} {split} dataset to 500 images")
        else:
            self.image_paths = [os.path.join(data_dir, d[i][0].strip("/")) for i in d.keys()]
            self.label_paths = [os.path.join(data_dir, d[i][1].strip("/")) for i in d.keys()]

    def _set_dataset_stat(self):
        self.target_label = 0
        if self.data in ('autopet', 'autopet_lung'):
            self.pet_intensity_range = (1.04, 51.21)
            self.pet_global_mean     = 7.06
            self.pet_global_std      = 7.96
        elif self.data == 'brats':
            # Intensity stats unused: BraTS uses the same percentile pipeline as AutoPET in _get_transforms.
            self.pet_intensity_range = (0.0, 1.0)
            self.pet_global_mean     = 0.0
            self.pet_global_std      = 1.0
        elif self.data == 'colon':
            self.intensity_range, self.global_mean, self.global_std = (-57, 175), 65.175035, 32.651197
        elif self.data == 'pancreas':
            self.intensity_range, self.global_mean, self.global_std = (-39, 204), 68.45214, 63.422806
            self.target_label = 2
        elif self.data == 'lits':
            self.intensity_range, self.global_mean, self.global_std = (-48, 163), 60.057533, 40.198017
            self.target_label = 2
        elif self.data == 'kits':
            self.intensity_range, self.global_mean, self.global_std = (-54, 247), 59.53867, 55.457336
            self.target_label = 2
        else:
            raise ValueError(f"Unknown dataset name: '{self.data}'. "
                             "Expected one of: autopet, autopet_lung, brats, colon, pancreas, lits, kits. "
                             "For HECKTOR use Dataset_hecktor.")

    def _get_transforms(self, split):
        if self.data in ('autopet', 'autopet_lung'):
            if split == "train":
                return Compose([
                    ChannelWisePercentileScale(keys=["image"], lower=0.05, upper=99.95,
                                               b_min=0.0, b_max=1.0, clip=True, debug=True),
                    RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=(128, 128, 128),
                                           label_key="label", pos=2, neg=0, num_samples=1,
                                           allow_smaller=True),
                    DivisiblePadd(keys=["image", "label"], k=32, value=0),
                    RandZoomd(keys=["image", "label"], prob=0.8, max_zoom=1.25,
                              mode=["trilinear", "nearest"]),
                ])
            else:
                return Compose([
                    ChannelWisePercentileScale(keys=["image"], lower=0.05, upper=99.95,
                                               b_min=0.0, b_max=1.0, clip=True),
                ])

        if self.data == 'brats':
            if split == "train":
                return Compose([
                    ChannelWisePercentileScale(keys=["image"], lower=0.05, upper=99.95,
                                               b_min=0.0, b_max=1.0, clip=True, debug=False),
                    RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=(128, 128, 128),
                                           label_key="label", pos=2, neg=0, num_samples=1,
                                           allow_smaller=True),
                    DivisiblePadd(keys=["image", "label"], k=32, value=0),
                    RandZoomd(keys=["image", "label"], prob=0.8, max_zoom=1.25,
                              mode=["trilinear", "nearest"]),
                ])
            else:
                return Compose([
                    ChannelWisePercentileScale(keys=["image"], lower=0.05, upper=99.95,
                                               b_min=0.0, b_max=1.0, clip=True),
                ])

        # CT datasets
        if split == "train":
            return Compose([
                ScaleIntensityRanged(keys=["image"],
                                     a_min=self.intensity_range[0], a_max=self.intensity_range[1],
                                     b_min=self.intensity_range[0], b_max=self.intensity_range[1],
                                     clip=True),
                RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=(128, 128, 128),
                                       label_key="label", pos=2, neg=0, num_samples=1),
                RandShiftIntensityd(keys=["image"], offsets=20, prob=0.5),
                NormalizeIntensityd(keys=["image"], subtrahend=self.global_mean, divisor=self.global_std),
                RandZoomd(keys=["image", "label"], prob=0.8, min_zoom=0.85, max_zoom=1.25,
                          mode=["trilinear", "nearest"]),
            ])
        else:
            return Compose([
                ScaleIntensityRanged(keys=["image"],
                                     a_min=self.intensity_range[0], a_max=self.intensity_range[1],
                                     b_min=self.intensity_range[0], b_max=self.intensity_range[1],
                                     clip=True),
                NormalizeIntensityd(keys=["image"], subtrahend=self.global_mean, divisor=self.global_std),
            ])

    def _binary_label(self, subject):
        subject.label.data = (subject.label.data == self.target_label).float()
        return subject

    def _pcc(self, subject):
        print("using pcc setting")
        random_index = torch.argwhere(subject.label.data == 1)
        if len(random_index) >= 1:
            random_index = random_index[np.random.randint(0, len(random_index))]
            crop_mask = torch.zeros_like(subject.label.data)
            crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
            subject.add_image(tio.LabelMap(tensor=crop_mask, affine=subject.label.affine),
                              image_name="crop_mask")
            subject = tio.CropOrPad(mask_name='crop_mask', target_shape=self.image_size)(subject)
        return subject


# ---------------------------------------------------------------------------
# Dataset_hecktor — PET/NPZ-based head-and-neck dataset (HECKTOR)
# ---------------------------------------------------------------------------

class Dataset_hecktor(Dataset):
    """
    Dataset for HECKTOR head-and-neck PET segmentation.

    Each .npz file contains:
      - 'pet': float16, shape (H, W, Z), already normalised to [0, 1]
      - 'lab': uint8,   shape (H, W, Z), binary {0, 1}

    Imbalance strategy (training only):
      RandCropByPosNegLabeld(pos=2, neg=0) guarantees every crop contains
      foreground voxels.  Voxel-level class imbalance is handled by
      DiceCELoss in trainer_basic.py.

    Val / test:
      Full volume, cropped tightly around the label with CropOrPad so the
      network always receives a (image_size)^3 patch that includes the tumour.
      No augmentation.  No balancing.
    """

    def __init__(self, data_dir, split='train', image_size=128, args=None):
        self.data_dir   = data_dir
        self.split      = split
        self.image_size = (image_size, image_size, image_size)
        self.args       = args

        split_file = os.path.join(data_dir, 'split.pkl')
        with open(split_file, 'rb') as f:
            d = pickle.load(f)[0][split]
        self.npz_paths = [
            os.path.join(data_dir, d[i][0].strip('/')) for i in d.keys()
        ]

        self.monai_transforms = self._get_transforms(split)

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, index):
        npz = np.load(self.npz_paths[index], allow_pickle=True)

        pet = torch.from_numpy(npz['pet'].astype(np.float32)).unsqueeze(0)  # (1, H, W, Z)
        lab = torch.from_numpy(npz['lab'].astype(np.float32)).unsqueeze(0)

        if lab.sum() == 0 and self.split == 'train':
            return self.__getitem__(np.random.randint(len(self)))

        if self.split == 'train':
            trans_dict = self.monai_transforms({'image': pet, 'label': lab})[0]
            return (
                trans_dict['image'].float(),
                trans_dict['label'].float(),
                self.npz_paths[index],
            )
        else:
            crop_fn = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=pet),
                label=tio.LabelMap(tensor=lab),
            )
            subject_save = tio.Subject(
                image=tio.ScalarImage(tensor=pet.clone()),
                label=tio.LabelMap(tensor=lab.clone()),
            )
            subject      = crop_fn(subject)
            subject_save = crop_fn(subject_save)

            trans_dict = self.monai_transforms({'image': subject.image.data})
            img_aug    = trans_dict['image']

            return (
                img_aug.float(),
                subject.label.data.float(),
                self.npz_paths[index],
                subject_save,
            )

    def _get_transforms(self, split):
        if split == 'train':
            return Compose([
                SpatialPadd(keys=['image', 'label'], spatial_size=self.image_size, mode='constant'),
                RandCropByPosNegLabeld(keys=['image', 'label'], spatial_size=self.image_size,
                                       label_key='label', pos=2, neg=0, num_samples=1),
                NormalizeIntensityd(keys=['image']),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
                RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
                RandZoomd(keys=['image', 'label'], prob=0.8, min_zoom=0.85, max_zoom=1.25,
                          mode=['trilinear', 'nearest']),
            ])
        else:
            return Compose([NormalizeIntensityd(keys=['image'])])


# ---------------------------------------------------------------------------
# Collate functions for lits / kits validation
# ---------------------------------------------------------------------------

def collate_fn_lits(batch):
    """Pass-through collate for dict-format val/test (batch_size=1).

    Required for LiTS and --full_volume (colon/pancreas/kits): default_collate
    would stack the nested list-of-tensors into 5D and break torchio Subject.
    """
    return batch[0]


def collate_fn_kits(batch):
    """Add batch dimension and verify spatial size for kits validation (batch_size=1).

    Only used for the legacy tight-crop kits path (not --full_volume).
    """
    image, label, image_path, subject_save = batch[0]
    image = image.unsqueeze(0)
    label = label.unsqueeze(0)
    expected = (128, 128, 128)
    if tuple(image.shape[-3:]) != expected:
        print(f"WARNING: kits collate — image size {tuple(image.shape[-3:])} != {expected}, resizing.")
        image = torch.nn.functional.interpolate(image, size=expected, mode='trilinear', align_corners=False)
    if tuple(label.shape[-3:]) != expected:
        label = torch.nn.functional.interpolate(label, size=expected, mode='nearest', align_corners=None)
    return image, label, image_path, subject_save


def _uses_sliding_window_dict(dataset) -> bool:
    """True when val/test samples are returned as LiTS-style subject dicts."""
    if getattr(dataset, 'data', None) == 'lits':
        return True
    args = getattr(dataset, 'args', None)
    return bool(getattr(args, 'full_volume', False))


# ---------------------------------------------------------------------------
# Dataloader_promise
# ---------------------------------------------------------------------------

class Dataloader_promise(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 num_workers=0, pin_memory=False, **kwargs):
        collate_fn = None
        if hasattr(dataset, 'data') and getattr(dataset, 'split', 'train') != 'train':
            if _uses_sliding_window_dict(dataset):
                collate_fn = collate_fn_lits
            elif dataset.data == 'kits':
                collate_fn = collate_fn_kits

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            **kwargs
        )

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
