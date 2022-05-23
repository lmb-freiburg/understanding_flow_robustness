"""
We edited this file.

Authors: Alex Wong <alexw@cs.ucla.edu>, Mukund Mundhra <mukundmundhra@cs.ucla.edu>

If this code is useful to you, please consider citing the following paper:

A. Wong, M. Mundhra, and S. Soatto. Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.
https://arxiv.org/pdf/2009.10142.pdf

@inproceedings{wong2021stereopagnosia,
  title={Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations},
  author={Wong, Alex and Mundhra, Mukund and Soatto, Stefano},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  year={2021}
}
"""
import argparse

import imagecorruptions

import global_attacks.global_constants as settings
from global_attacks.perturb_main import run

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="kitti2015",
    choices=["kitti2012", "kitti2015"],
)
# Run settings
parser.add_argument(
    "--n_height", type=int, default=settings.N_HEIGHT, help="Height of each sample"
)
parser.add_argument(
    "--n_width", type=int, default=settings.N_WIDTH, help="Width of each sample"
)
# Perturb model settings
parser.add_argument(
    "--perturb_method",
    type=str,
    default=settings.PERTURB_METHOD,
    choices=settings.PERTURB_METHOD_AVAILABLE + imagecorruptions.get_corruption_names(),
)
parser.add_argument(
    "--perturb_mode",
    type=str,
    default="both",
    help="Perturb modes available: %s" % settings.PERTURB_MODE_AVAILABLE,
)
parser.add_argument(
    "--output_norm", default=settings.OUTPUT_NORM, help="Output norm of noise"
)
parser.add_argument(
    "--n_step",
    type=int,
    default=settings.N_STEP,
    help="Number of steps to optimize perturbations",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-3,
    help="Learning rate (alpha) to use for optimizing perturbations",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.47,
    help="Momentum (beta) used for momentum iterative fast gradient sign method",
)
parser.add_argument(
    "--probability_diverse_input",
    type=float,
    default=0.00,
    help="Probability (p) to use diverse input",
)
# Stereo model settings
parser.add_argument(
    "--disparity_method",
    type=str,
    default=settings.STEREO_METHOD,
    help="Stereo method available: %s" % settings.STEREO_METHOD_AVAILABLE,
)
parser.add_argument("--flownet", type=str, default="FlowNetC", help="Flownet model")
parser.add_argument(
    "--disparity_model_restore_path",
    type=str,
    default="",
    help="Path to restore model checkpoint",
)
# Output settings
parser.add_argument(
    "--output_path", type=str, default=settings.OUTPUT_PATH, help="Path to save outputs"
)
# Hardware settings
parser.add_argument(
    "--device", type=str, default=settings.DEVICE, help="Device to use: gpu, cpu"
)
parser.add_argument("--disparity", action="store_true", help="Run disparity estimation")
parser.add_argument("--DEBUG", action="store_true", help="Debug mode")
parser.add_argument(
    "--write_out", action="store_true", help="Write out qualitative results"
)
parser.add_argument(
    "--write_out_npy",
    action="store_true",
    help="Write out qualitative results as npy too",
)
parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="Set seed. Note that this might overwrite previous results with this seed! If not specified unused seed is used",
)

# Only useful for white-box adv. attacks
parser.add_argument(
    "--flow_loss",
    type=str,
    default="l2",
    choices=["cossim", "l2", "l1", "corr"],
    help="What type of flow loss to use",
)
parser.add_argument("--targeted", action="store_true", help="Create targeted examples")
parser.add_argument(
    "--show_evolve", action="store_true", help="Show evolution of perturbation"
)
parser.add_argument(
    "--homogeneous", action="store_true", help="Show evolution of perturbation"
)
parser.add_argument(
    "--arbitrary_gt_index",
    default=None,
    help="Perturb all images with arbitrary same GT",
)
parser.add_argument(
    "--arbitrary_noise_index",
    default=None,
    type=int,
    help="Perturb all images with arbitrary same noise perturbation",
)

# only useful for universal perturbation attacks
parser.add_argument(
    "--universal_evaluation",
    action="store_true",
    help="Run universal evaluation pipeline",
)
parser.add_argument(
    "--folder_name",
    type=str,
    default="",
)
parser.add_argument(
    "--epoch_number",
    type=int,
    default=-1,
)
parser.add_argument(
    "--uniform_noise",
    action="store_true",
    help="Run universal evaluation pipeline with uniform noise",
)

args = parser.parse_args()


def isfloat(x) -> bool:
    try:
        _ = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x) -> bool:
    try:
        a = float(x)
        b = int(a)
    except (TypeError, ValueError):
        return False
    else:
        return a == b


if __name__ == "__main__":
    args.perturb_method = args.perturb_method.lower()
    args.perturb_mode = args.perturb_mode.lower()
    args.disparity_method = args.disparity_method.lower()
    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    if args.universal_evaluation:
        args.return_feat_maps = False
        args.show_evolve = False
        args.homogeneous = False
        args.targeted = False
        args.arbitrary_gt_index = None
        args.arbitrary_noise_index = None
        run(
            # Run settings
            n_height=args.n_height,
            n_width=args.n_width,
            # Perturb settings
            perturb_method=args.perturb_method,
            perturb_mode=args.perturb_mode,
            output_norm=args.output_norm,
            # Stereo model settings
            stereo_method=args.disparity_method,
            stereo_model_restore_path=args.disparity_model_restore_path,
            # Output settings
            output_path=args.output_path,
            # Hardware settings
            device=args.device,
            args=args,
        )
    elif args.perturb_method in settings.PERTURB_METHOD_AVAILABLE:
        if isint(args.output_norm):
            args.output_norm = int(args.output_norm)
        elif isfloat(args.output_norm):
            args.output_norm = float(args.output_norm)
        args.return_feat_maps = True if "corr" == args.flow_loss else False
        run(
            # Run settings
            n_height=args.n_height,
            n_width=args.n_width,
            # Perturbations Settings
            perturb_method=args.perturb_method,
            perturb_mode=args.perturb_mode,
            output_norm=args.output_norm,
            n_step=args.n_step,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            probability_diverse_input=args.probability_diverse_input,
            # Stereo model settings
            stereo_method=args.disparity_method,
            stereo_model_restore_path=args.disparity_model_restore_path,
            # Output settings
            output_path=args.output_path,
            # Hardware settings
            device=args.device,
            args=args,
        )
    elif args.perturb_method in imagecorruptions.get_corruption_names():
        args.return_feat_maps = False
        args.show_evolve = False
        args.homogeneous = False
        args.targeted = False
        args.arbitrary_gt_index = None
        args.arbitrary_noise_index = None
        for severity in range(5):
            run(
                # Run settings
                n_height=args.n_height,
                n_width=args.n_width,
                # Perturbations Settings
                perturb_method=args.perturb_method,
                perturb_mode=args.perturb_mode,
                output_norm=severity + 1,
                n_step=args.n_step,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                probability_diverse_input=args.probability_diverse_input,
                # Stereo model settings
                stereo_method=args.disparity_method,
                stereo_model_restore_path=args.disparity_model_restore_path,
                # Output settings
                output_path=args.output_path,
                # Hardware settings
                device=args.device,
                args=args,
            )
    else:
        raise NotImplementedError
