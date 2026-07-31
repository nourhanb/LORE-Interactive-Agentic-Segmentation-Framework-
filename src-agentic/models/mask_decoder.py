import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Union, Optional, Dict, Any
from .unet import Unet_decoder, Conv, TwoConv
from monai.networks.nets import UNet


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MEDSA Spatial Next-Prompt Head
# ─────────────────────────────────────────────────────────────────────────────

class SpatialNextPromptHead(nn.Module):
    """
    Predicts a 3D heatmap H_i ∈ [0,1]^{D×H×W} indicating where the physician
    should correct next.  Conditioned on the spatial decoder features via two
    FiLM modulations from the edit embedding e_i_full and memory summary m_i.

    Architecture:
        FiLM(e_i_full) → residual block 1 → FiLM(m_i) → residual block 2
        → 1×1×1 conv → sigmoid
    """
    EDIT_DIM = 132   # EditDeltaEncoder.FULL_DIM

    def __init__(
        self,
        spatial_channels: int = 32,
        act:  Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
    ):
        super().__init__()
        C = spatial_channels

        # FiLM γ/β projections from edit embedding
        self.film_e_gamma = nn.Linear(self.EDIT_DIM, C)
        self.film_e_beta  = nn.Linear(self.EDIT_DIM, C)

        # FiLM γ/β projections from memory summary
        self.film_m_gamma = nn.Linear(self.EDIT_DIM, C)
        self.film_m_beta  = nn.Linear(self.EDIT_DIM, C)

        self.res_block1 = TwoConv(3, C, C, act, norm, bias=True, dropout=0.0)
        self.res_block2 = TwoConv(3, C, C, act, norm, bias=True, dropout=0.0)

        self.out_conv = Conv["conv", 3](in_channels=C, out_channels=1, kernel_size=1)

    def forward(
        self,
        feat:   torch.Tensor,        # (B, C, X, Y, Z) — upscaled decoder features
        e_full: torch.Tensor,        # (B, 132) — edit embedding
        m_i:    torch.Tensor,        # (B, 132) — memory summary
        f_persistent: bool = False,
        persist_region: Optional[torch.Tensor] = None,  # (B,1,X,Y,Z) amplify mask
    ) -> torch.Tensor:               # (B, 1, X, Y, Z) heatmap in [0,1]

        # ── FiLM conditioning from edit embedding ─────────────────────────
        gamma_e = self.film_e_gamma(e_full).view(-1, feat.size(1), 1, 1, 1)
        beta_e  = self.film_e_beta(e_full).view(-1, feat.size(1), 1, 1, 1)
        x = (1 + gamma_e) * feat + beta_e

        # First residual block
        x = x + self.res_block1(x)

        # ── FiLM conditioning from memory summary ─────────────────────────
        gamma_m = self.film_m_gamma(m_i).view(-1, feat.size(1), 1, 1, 1)
        beta_m  = self.film_m_beta(m_i).view(-1, feat.size(1), 1, 1, 1)
        x = (1 + gamma_m) * x + beta_m

        # Second residual block
        x = x + self.res_block2(x)

        heatmap = torch.sigmoid(self.out_conv(x))   # (B,1,X,Y,Z)

        # Amplify persistent-failure region ×2 (clipped to 1)
        if f_persistent and persist_region is not None:
            heatmap = (heatmap * (1.0 + persist_region)).clamp(0.0, 1.0)

        return heatmap


class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        args,
        *,
        transformer_dim: int = 384,
        multiple_outputs: bool = False,
        num_multiple_outputs: int = 3,
    ) -> None:
        super().__init__()
        self.args = args
        self.multiple_outputs = multiple_outputs
        self.num_multiple_outputs = num_multiple_outputs
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, 32, 3) for i in range(num_multiple_outputs + 1)])
        self.iou_prediction_head = MLP(transformer_dim, 256, num_multiple_outputs + 1, 3, sigmoid_output=True)

        self.decoder = Unet_decoder(spatial_dims=3, features=(32, 32, 64, 128, transformer_dim, 32))

        # ── MEDSA Spatial Next-Prompt Head ──────────────────────────────────
        if getattr(self.args, 'use_medsa', False):
            self.spatial_head = SpatialNextPromptHead(spatial_channels=32)

        if self.args.refine:
            self.refine = Refine(self.args)

    def forward(
        self,
        prompt_embeddings: torch.Tensor,          # (B, num_mask_tokens, C)
        image_embeddings:  torch.Tensor,          # (B, C, X/4, Y/4, Z/4)
        feature_list:      List[torch.Tensor],
        medsa_context:     Optional[Dict[str, Any]] = None,
    ) -> Tuple:
        """
        Returns:
            masks        (B, N, X, Y, Z)  — segmentation logits per candidate
            iou_pred     (B, N)            — predicted quality / confidence
            spatial_hmap (B, 1, X, Y, Z) or None — MEDSA next-prompt heatmap
            iou_token    (B, C)            — global context token (for policy)
        """
        upscaled_embedding = self.decoder(image_embeddings, feature_list)
        masks, iou_pred, iou_token = self._predict_mask(
            upscaled_embedding, prompt_embeddings
        )

        spatial_hmap = None
        if getattr(self.args, 'use_medsa', False) and medsa_context is not None:
            e_full = medsa_context.get('e_full')
            m_i    = medsa_context.get('m_i')
            if e_full is not None and m_i is not None:
                spatial_hmap = self.spatial_head(
                    upscaled_embedding,
                    e_full,
                    m_i,
                    f_persistent=medsa_context.get('f_persistent', False),
                    persist_region=medsa_context.get('persist_region', None),
                )

        return masks, iou_pred, spatial_hmap, iou_token


    def _predict_mask(self, upscaled_embedding, prompt_embeddings):
        b, c, x, y, z = upscaled_embedding.shape

        iou_token_out   = prompt_embeddings[:, 0, :]
        mask_tokens_out = prompt_embeddings[:, 1: (self.num_multiple_outputs + 1 + 1), :]

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_multiple_outputs + 1):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        masks       = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.multiple_outputs:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks    = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # iou_token_out also exposed for the MEDSA policy state vector
        return masks, iou_pred, iou_token_out

class Refine_unet(nn.Module):
    def __init__(self):
        super(Refine_unet, self).__init__()
        self.refine = UNet(spatial_dims=3, in_channels=4, out_channels=1,
                           channels=(32, 64, 64), strides=(2, 2), num_res_units=2)
    def forward(self, x):
        return self.refine(x)

class Refine(nn.Module):
    def __init__(self,
                 args,
                 spatial_dims: int = 3,
                 in_channel: int = 4,
                 out_channel: int = 32,
                 act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm: Union[str, tuple] = ("instance", {"affine": True}),
                 bias: bool = True,
                 dropout: Union[float, tuple] = 0.0,
                 ):
        super().__init__()
        self.args = args

        self.first_conv = Conv["conv", 3](in_channels=in_channel, out_channels=out_channel, kernel_size=1)

        self.conv1 = TwoConv(spatial_dims, out_channel, out_channel, act, norm, bias, dropout)
        self.conv2 = TwoConv(spatial_dims, out_channel, out_channel, act, norm, bias, dropout)

        self.conv_error_map = Conv["conv", 3](in_channels=out_channel, out_channels=1, kernel_size=1)
        self.conv_correction = Conv["conv", 3](in_channels=out_channel, out_channels=1, kernel_size=1)


    def forward(self, image, mask_best, points, mask):

        x = self._get_refine_input(image, mask_best, points)
        mask = F.interpolate(mask, scale_factor=0.5, mode='trilinear', align_corners=False)

        x = self.first_conv(x)

        residual = x
        x = self.conv1(x)
        x = residual + x

        residual = x
        x = self.conv2(x)
        x = residual + x

        error_map = self.conv_error_map(x)
        correction = self.conv_correction(x)

        outputs = (error_map * correction + mask)

        outputs = F.interpolate(outputs, scale_factor=2, mode='trilinear', align_corners=False)
        error_map = F.interpolate(error_map, scale_factor=2, mode='trilinear', align_corners=False)
        return outputs, error_map

    def _get_refine_input(self, image, mask, points):
        mask = torch.sigmoid(mask)
        mask = (mask > 0.5)

        coors, labels = points[0], points[1]
        positive_map, negative_map = torch.zeros_like(image), torch.zeros_like(image)

        for click_iters in range(len(coors)):
            coors_click, labels_click = coors[click_iters], labels[click_iters]
            for batch in range(image.size(0)):
                point_label = labels_click[batch]
                coor = coors_click[batch]
                # sepehre_coor = [coor[:, 0], coor[:, 1], coor[:, 2]]

                # Create boolean masks
                negative_mask = point_label == 0
                positive_mask = point_label != 0

                # Update negative_map
                if negative_mask.any():  # Check if there's at least one True in negative_mask
                    negative_indices = coor[negative_mask]
                    for idx in negative_indices:
                        negative_map[batch, 0, idx[0], idx[1], idx[2]] = 1

                # Update positive_map
                if positive_mask.any():  # Check if there's at least one True in negative_mask
                    positive_indices = coor[positive_mask]
                    for idx in positive_indices:
                        positive_map[batch, 0, idx[0], idx[1], idx[2]] = 1


        refine_input = F.interpolate(torch.cat([image, mask, positive_map, negative_map], dim=1), scale_factor=0.5, mode='trilinear')
        return refine_input

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

