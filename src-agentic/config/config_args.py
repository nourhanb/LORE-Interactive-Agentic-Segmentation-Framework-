import argparse
import os
import warnings
parser = argparse.ArgumentParser()


# data
parser.add_argument("--data", default=None, type=str,
                    choices=["kits", "pancreas", "lits", "colon", "hecktor", "autopet", "brats"])
parser.add_argument("--save_dir", default="./implementation/", type=str)
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--split", default="train", type=str)
parser.add_argument('--use_small_dataset', action="store_true")


# network
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument("--lr", default=4e-5, type=float)
parser.add_argument("--lr_scheduler", default='linear', type=str, choices=["linear", "exp"])
parser.add_argument('--warm_up', action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--max_epoch", default=200, type=int)
parser.add_argument("--image_size", default=128, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--checkpoint", default="best", type=str)
parser.add_argument("--checkpoint_sam", default="./checkpoint_sam/sam_vit_b_01ec64.pth", type=str,
                    help='path of pretrained SAM')
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--tolerance", default=5, type=int)
parser.add_argument("--boundary_kernel_size", default=5, type=int,
                    help='an integer for kernel size of avepooling layer for boundary generation')
parser.add_argument("--use_pretrain", action="store_true")
parser.add_argument("--pretrain_path", default="", type=str)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_best", action="store_true")
parser.add_argument("--ddp", action="store_true")
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--accumulation_steps', type=int, default=20)

parser.add_argument('--iter_nums', type=int, default=11)
parser.add_argument('--num_clicks', type=int, default=50)
parser.add_argument('--num_clicks_validation', type=int, default=10)
parser.add_argument('--use_box', action="store_true")
parser.add_argument('--dynamic_box', action="store_true")
parser.add_argument('--use_scribble', action="store_true")


parser.add_argument('--num_multiple_outputs', type=int, default=3)
parser.add_argument('--multiple_outputs', action="store_true")
parser.add_argument('--refine', action="store_true")
parser.add_argument('--no_detach', action="store_true")
parser.add_argument('--refine_test', action="store_true")

parser.add_argument('--dynamic', action="store_true")
parser.add_argument('--efficient_scribble', action="store_true")
parser.add_argument("--use_sam3d_turbo", action="store_true")
parser.add_argument("--init_learning", action="store_true")



# saving
parser.add_argument("--save_predictions", action="store_true")
parser.add_argument("--save_csv", action="store_true")
parser.add_argument("--save_test_dir", default='./', type=str)
parser.add_argument("--save_name", default='testing_only', type=str)






# ── Clinical utility heads ─────────────────────────────────────────────────
# Acceptance head: predicts whether a physician would accept the mask
parser.add_argument('--accept_dice_threshold', type=float, default=0.90,
                    help='Dice >= threshold → physician accept label = 1')
parser.add_argument('--lambda_accept', type=float, default=0.0,
                    help='Weight for acceptance BCE loss in total loss (0 = disabled)')

# Effort head: predicts physician correction effort
parser.add_argument('--lambda_effort', type=float, default=0.0,
                    help='Weight for effort L1 loss in total loss (0 = disabled)')
parser.add_argument('--effort_target_type', type=str, default='assd',
                    choices=['voxel_error', 'boundary_error', 'assd'],
                    help='Proxy used to simulate physician correction effort. '
                         '"assd" = normalised average symmetric surface distance '
                         '(geometrically independent from Dice). '
                         '"voxel_error" = mean abs voxel discrepancy. '
                         '"boundary_error" = (FP+FN)/(|pred|+|gt|).')

# Candidate mask selector
parser.add_argument('--selector_mode', type=str, default='confidence',
                    choices=['confidence', 'acceptance', 'acceptance_effort'],
                    help='Score used to pick the best candidate mask. '
                         '"confidence" preserves original behaviour.')
parser.add_argument('--selector_alpha', type=float, default=1.0,
                    help='accept_prob weight in utility = alpha*accept - beta*effort + gamma*quality')
parser.add_argument('--selector_beta', type=float, default=0.5,
                    help='effort_pred weight in utility score')
parser.add_argument('--selector_gamma', type=float, default=1.0,
                    help='confidence (iou) weight in utility score')

# Early stopping
parser.add_argument('--enable_early_stop', action='store_true',
                    help='Stop iterative correction early when acceptance exceeds threshold')
parser.add_argument('--stop_accept_threshold', type=float, default=0.95,
                    help='Mean acceptance prob above which to stop iterating (inference only)')


# ── MEDSA: Mask-Edit Driven Spatial Agent ──────────────────────────────────
parser.add_argument('--use_medsa', action='store_true',
                    help='Enable MEDSA agentic block (EditDeltaEncoder + Memory + Spatial Head + DQN)')
parser.add_argument('--force_action', type=str, default=None,
                    choices=['point', 'box', 'scribble'],
                    help='Override the policy and use this fixed prompt type for every interaction step. '
                         'Disables adaptive stopping. Useful for ablation comparisons.')
parser.add_argument('--fixed_k', type=int, default=-1,
                    help='When --force_action is set, stop after exactly this many interactions '
                         'instead of iter_nums. Use the dataset\'s mean K to match the agent budget.')

# Spatial Next-Prompt Head
parser.add_argument('--spatial_sigma', type=float, default=5.0,
                    help='Gaussian sigma (voxels) for GT spatial heatmap supervision')

# Curriculum noise on edit deltas
parser.add_argument('--medsa_noise_sigma_start', type=float, default=15.0,
                    help='Initial noise sigma for edit delta curriculum (voxels). '
                         'Anneals to 0 by epoch medsa_noise_anneal_end.')
parser.add_argument('--medsa_noise_anneal_end', type=int, default=150,
                    help='Epoch at which curriculum noise reaches zero.')

# Loss weight warmup
parser.add_argument('--medsa_spatial_ramp_end', type=int, default=80,
                    help='Epoch at which lambda_spatial reaches its max value (0.5).')
parser.add_argument('--medsa_rl_start_epoch', type=int, default=50,
                    help='Epoch at which DQN loss begins to ramp up.')
parser.add_argument('--medsa_rl_ramp_len', type=int, default=100,
                    help='Number of epochs over which lambda_rl ramps from 0 to 1.')

# DQN hyperparameters
parser.add_argument('--dqn_replay_capacity', type=int, default=10_000,
                    help='Experience replay buffer capacity.')
parser.add_argument('--dqn_batch_size', type=int, default=64,
                    help='Mini-batch size for DQN updates.')
parser.add_argument('--dqn_gamma', type=float, default=0.99,
                    help='Discount factor for DQN.')
parser.add_argument('--dqn_target_update_freq', type=int, default=100,
                    help='Steps between soft target network updates.')

# MEDSA optimiser
parser.add_argument('--medsa_lr', type=float, default=1e-4,
                    help='Learning rate for MEDSA policy components '
                         '(edit encoder, memory, spatial head, DQN).')

parser.add_argument('--use_clinical_priority', action='store_true',
                    help='(Legacy/MedSA hybrid) Rank FN/FP CCs by clinical priority.')
parser.add_argument('--use_uncertainty_gate', action='store_true',
                    help='(Legacy/MedSA hybrid) Entropy/persistence escalate vs accept.')
parser.add_argument('--uncert_low', type=float, default=0.35,
                    help='Uncertainty threshold below which auto-accept is allowed.')
parser.add_argument('--uncert_high', type=float, default=0.65,
                    help='Uncertainty threshold above which escalate is forced.')
parser.add_argument('--prio_lam_iso', type=float, default=1.0,
                    help='Weight of isolation term in clinical priority.')
parser.add_argument('--prio_lam_int', type=float, default=0.5,
                    help='Weight of intensity-contrast term in clinical priority.')
parser.add_argument('--prio_lam_bnd', type=float, default=0.5,
                    help='Weight of boundary term in clinical priority.')
parser.add_argument('--beta_priority', type=float, default=0.30,
                    help='Reward weight for priority_hit.')
parser.add_argument('--gamma_uncert', type=float, default=0.20,
                    help='Reward weight for uncertainty escalate/accept terms.')
parser.add_argument('--persist_thresh', type=float, default=0.85,
                    help='(Legacy MedSA) Persistence cosine threshold for escalate.')

# ── Full-FOV / no tight 128³ tumor crop (SPIE clinical figures) ─────────────
parser.add_argument('--full_volume', action='store_true',
                    help='Disable tight label-centered CropOrPad(128) on val/test. '
                         'Return the full (or context-expanded) CT and evaluate with '
                         'sliding-window 128³ inference (LiTS-style). Train still uses '
                         'random 128³ patches from the uncropped volume.')
parser.add_argument('--context_margin', type=int, default=96,
                    help='When --full_volume is set on very large volumes (e.g. KiTS), '
                         'expand the label bbox by this many voxels on each side instead '
                         'of loading the entire CT. Colon/pancreas use the true full volume.')


def use_sliding_window(args) -> bool:
    """Val/test should stitch 128³ patches over a full/context volume."""
    if getattr(args, 'data', None) == 'lits':
        return True
    return bool(getattr(args, 'full_volume', False))


def check_and_setup_parser(args):
    if args.save_name == 'testing_only':
        warnings.warn("[save_name] (--save_name) should be a real name, currently is for testing purpose (--save_name=testing_only)")


    args.save_dir = os.path.join(args.save_dir, args.data, args.save_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)