from dataset_utils import custom_transforms
from dataset_utils.validation_flow import ValidationFlowKitti2012, ValidationFlowKitti2015
from dataset_utils.validation_sintel import MpiSintel


def get_evaluation_set(flow_loader_h, flow_loader_w, args):
    valid_transform = custom_transforms.Compose(
        [
            custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
            custom_transforms.ArrayToTensor(),
        ]
    )

    if args.valset == "kitti2015":
        val_set = ValidationFlowKitti2015(
            root="datasets/KITTI/2015",
            transform=valid_transform,
            compression=args.compression if "compression" in args else 0,
            raw_root="datasets/KITTI/raw",
            example=args.example if "example" in args else 0,
            true_motion=args.true_motion if "true_motion" in args else False,
        )
    elif args.valset == "kitti2012":
        val_set = ValidationFlowKitti2012(
            root="datasets/KITTI/2012",
            transform=valid_transform,
            compression=args.compression if "compression" in args else None,
        )
    elif args.valset == "sintel":
        val_set = MpiSintel(
            root="datasets/Sintel",
            transform=valid_transform,
            split="training",
            dstype="clean",
        )
    return val_set
