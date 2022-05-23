import re

import numpy as np
import torch
from path import Path

import models


def get_flownet_choices():
    return [
        "FlowNetS",
        "FlowNetC",
        "FlowNet2",
        "FlowNetCFlexLarger_k3_reps3",  # Robust FlowNetC
        "FlowNetCFlexLarger_k3_reps3_adv_ifgsm_l2_002",
        "FlowNetCFlexLarger_k5_reps0",  # Original FlowNetC trained with our pipeline
        "SpyNet",
        "PWCNet",
        "PWCNet_adv_ifgsm_l2_002",
        "RAFT",
        "RAFT_FlowNetCEncoder_WoContext",
        "RAFT_adv_kitti2012_ifgsm_l2_002",
    ]


def fetch_model(
    args, pretrained_path: str = "pretrained_models", return_feat_maps: bool = False
) -> torch.nn.Module:
    """Create model and initialize with weights

    args:
        pretrained_path ([type]): [description]

    Returns:
        torch.model: flownet model with pretrained weights
    """
    pretrained_path = Path(pretrained_path)

    if args.flownet == "SpyNet":
        flow_net = getattr(models, args.flownet)(nlevels=6, pretrained=True)
    elif args.flownet == "PWCNet":
        flow_net = models.pwc_dc_net(
            pretrained_path / "pwc_net_chairs.pth.tar",
            return_feat_maps=return_feat_maps,
        )
    elif "PWCNet_adv" in args.flownet:
        flow_net = getattr(models, "PWCDCNet")()
    elif re.findall("^RAFT", args.flownet):
        args.small = False
        args.mixed_precision = False if "adv" in args.flownet else True
        args.alternate_corr = False
        args.fnorm = "instance"
        args.cnorm = "batch"
        args.no_separate_context = (
            True
            if "No_Separate_Context" in args.flownet
            or "FlowNetCEncoder_WoContext" in args.flownet
            else False
        )
        args.corr_levels = 4
        args.iters = 12
        args.flowNetCEnc = (
            True
            if "FlowNetCEncoder" in args.flownet
            or "FlowNetCEncoder_WoContext" in args.flownet
            else False
        )
        args.raft_parallel = False
        if args.raft_parallel:
            flow_net = torch.nn.DataParallel(
                models.raft.raft.RAFT(args, return_feat_maps=return_feat_maps)
            )
        else:
            flow_net = models.raft.raft.RAFT(args, return_feat_maps=return_feat_maps)
    else:
        if re.findall("^FlowNetCFlexLarger", args.flownet):
            kernel_size = 5 if re.findall("k5", args.flownet) else 3
            match = re.search("reps([0-3])", args.flownet)
            number_of_reps = int(match.group(0)[4:])
            if (
                kernel_size == 5
                and number_of_reps == 0
                and "dil" not in args.flownet
                and "relu" not in args.flownet
            ):
                flow_net = getattr(models, "FlowNetC_predict_bias")(
                    return_feat_maps=return_feat_maps
                )
            else:
                flow_net = getattr(models, "FlowNetC_flexible_larger_field")(
                    kernel_size=kernel_size,
                    number_of_reps=number_of_reps,
                    dilation=1,
                    return_feat_maps=return_feat_maps,
                )
        else:
            flow_net = getattr(models, args.flownet)(return_feat_maps=return_feat_maps)

    # load weights
    if args.flownet in ["SpyNet", "PWCNet"]:
        pass
    elif args.flownet == "FlowNetC":
        weights = torch.load(pretrained_path / "FlowNet2-C_checkpoint.pth.tar")
        flow_net.load_state_dict(weights["state_dict"])
    elif re.findall("^FlowNetCFlexLarger", args.flownet):
        if "k5_reps0" in args.flownet:
            weights = torch.load(pretrained_path / "larger_field_3x3_x0_l2.pth")
        elif "k3_reps3" in args.flownet:
            if "adv_ifgsm_l2_002" in args.flownet:
                weights = torch.load(
                    pretrained_path / "adv_kitti2012_robustFlow_ifgsm_l2_0.02.pth"
                )
            else:
                weights = torch.load(pretrained_path / "RobustFlowNetC.pth")
        flow_net.load_state_dict(weights)

    elif args.flownet in ["FlowNetS"]:
        print("=> using pre-trained weights for FlowNetS")
        weights = torch.load(pretrained_path / "FlowNet2-S_checkpoint.pth.tar")
        flow_net.load_state_dict(weights["state_dict"])
    elif re.findall("^RAFT", args.flownet):
        if "FlowNetCEncoder_WoContext" in args.flownet:
            weights = torch.load(
                pretrained_path / "raft_flowNetCEnc_noSeparateContext.pth"
            )
        elif "RAFT_adv_kitti2012_ifgsm_l2_002" == args.flownet:
            weights = torch.load(pretrained_path / "adv_kitti2012_raft_ifgsm_l2_0.02.pth")
        else:
            weights = torch.load(str(pretrained_path / "raft-things.pth"))

        try:
            flow_net.load_state_dict(weights)
        except Exception:
            count = 0
            model_state_dict = flow_net.state_dict()
            new = list(weights.items())
            for key, _ in model_state_dict.items():
                _, weights = new[count]
                model_state_dict[key] = weights
                count += 1
            flow_net.load_state_dict(model_state_dict)
    elif args.flownet in ["FlowNet2"]:
        print("=> using pre-trained weights for FlowNet2")
        weights = torch.load(pretrained_path / "FlowNet2_checkpoint.pth.tar")
        flow_net.load_state_dict(weights["state_dict"])
    elif "PWCNet_adv" in args.flownet:
        if args.flownet == "PWCNet_adv_ifgsm_l2_002":
            weights = torch.load(
                pretrained_path / "adv_kitti2012_pwcnet_ifgsm_l2_0.02.pth"
            )
        flow_net.load_state_dict(weights)
    else:
        print(f"Use init weights for {args.flownet}")
        flow_net.init_weights()

    return flow_net


def setup_hooks(flow_net, args):
    if args.flownet in ["SpyNet"]:
        raise NotImplementedError
    elif (
        args.flownet in ["FlowNetC", "FlowNetC_original", "FlowNetCFlexLarger_k3_reps3"]
        or "FlowNetCFlexLarger" in args.flownet
    ):
        flow_net.conv_redir.register_forward_hook(get_activation("conv_redir"))
        flow_net.conv3_1.register_forward_hook(get_activation("conv3_1"))
        flow_net.conv4.register_forward_hook(get_activation("conv4"))
        flow_net.conv4_1.register_forward_hook(get_activation("conv4_1"))
        flow_net.conv5.register_forward_hook(get_activation("conv5"))
        flow_net.conv5_1.register_forward_hook(get_activation("conv5_1"))
        flow_net.conv6.register_forward_hook(get_activation("conv6"))
        flow_net.conv6_1.register_forward_hook(get_activation("conv6_1"))
        flow_net.predict_flow6.register_forward_hook(get_activation("flow6"))
        flow_net.deconv5.register_forward_hook(get_activation("deconv5"))
        flow_net.predict_flow5.register_forward_hook(get_activation("flow5"))
        flow_net.deconv4.register_forward_hook(get_activation("deconv4"))
        flow_net.predict_flow4.register_forward_hook(get_activation("flow4"))
        flow_net.deconv3.register_forward_hook(get_activation("deconv3"))
        flow_net.predict_flow3.register_forward_hook(get_activation("flow3"))
        flow_net.deconv2.register_forward_hook(get_activation("deconv2"))
        flow_net.predict_flow2.register_forward_hook(get_activation("predict"))
        flow_net.upsampled_flow6_to_5.register_forward_hook(
            get_activation("upsampled_flow6_to_5")
        )
        flow_net.upsampled_flow5_to_4.register_forward_hook(
            get_activation("upsampled_flow5_to_4")
        )
        flow_net.upsampled_flow4_to_3.register_forward_hook(
            get_activation("upsampled_flow4_to_3")
        )
        flow_net.upsampled_flow3_to_2.register_forward_hook(
            get_activation("upsampled_flow3_to_2")
        )
    elif args.flownet in ["FlowNetS"]:
        flow_net.conv1.register_forward_hook(get_activation("conv1"))
        flow_net.conv2.register_forward_hook(get_activation("conv2"))
        flow_net.conv3.register_forward_hook(get_activation("conv3"))
        flow_net.conv3_1.register_forward_hook(get_activation("conv3_1"))
        flow_net.conv4.register_forward_hook(get_activation("conv4"))
        flow_net.conv4_1.register_forward_hook(get_activation("conv4_1"))
        flow_net.conv5.register_forward_hook(get_activation("conv5"))
        flow_net.conv5_1.register_forward_hook(get_activation("conv5_1"))
        flow_net.conv6.register_forward_hook(get_activation("conv6"))
        flow_net.conv6_1.register_forward_hook(get_activation("conv6_1"))
        flow_net.deconv5.register_forward_hook(get_activation("deconv5"))
        flow_net.deconv4.register_forward_hook(get_activation("deconv4"))
        flow_net.deconv3.register_forward_hook(get_activation("deconv3"))
        flow_net.deconv2.register_forward_hook(get_activation("deconv2"))
        flow_net.predict_flow6.register_forward_hook(get_activation("predict_flow6"))
        flow_net.predict_flow5.register_forward_hook(get_activation("predict_flow5"))
        flow_net.predict_flow4.register_forward_hook(get_activation("predict_flow4"))
        flow_net.predict_flow3.register_forward_hook(get_activation("predict_flow3"))
        flow_net.predict_flow2.register_forward_hook(get_activation("predict_flow2"))
        flow_net.upsampled_flow6_to_5.register_forward_hook(
            get_activation("upsampled_flow6_to_5")
        )
        flow_net.upsampled_flow5_to_4.register_forward_hook(
            get_activation("upsampled_flow5_to_4")
        )
        flow_net.upsampled_flow4_to_3.register_forward_hook(
            get_activation("upsampled_flow4_to_3")
        )
        flow_net.upsampled_flow3_to_2.register_forward_hook(
            get_activation("upsampled_flow3_to_2")
        )
    elif args.flownet in ["FlowNet2"]:
        raise NotImplementedError
    elif args.flownet in ["PWCNet"]:
        flow_net.conv6_0.register_forward_hook(get_activation("conv6_0"))
        flow_net.conv6_1.register_forward_hook(get_activation("conv6_1"))
        flow_net.conv6_2.register_forward_hook(get_activation("conv6_2"))
        flow_net.conv6_3.register_forward_hook(get_activation("conv6_3"))
        flow_net.conv6_4.register_forward_hook(get_activation("conv6_4"))
        flow_net.predict_flow6.register_forward_hook(get_activation("predict_flow6"))
        flow_net.deconv6.register_forward_hook(get_activation("deconv6"))
        flow_net.upfeat6.register_forward_hook(get_activation("upfeat6"))
        flow_net.conv5_0.register_forward_hook(get_activation("conv5_0"))
        flow_net.conv5_1.register_forward_hook(get_activation("conv5_1"))
        flow_net.conv5_2.register_forward_hook(get_activation("conv5_2"))
        flow_net.conv5_3.register_forward_hook(get_activation("conv5_3"))
        flow_net.conv5_4.register_forward_hook(get_activation("conv5_4"))
        flow_net.predict_flow5.register_forward_hook(get_activation("predict_flow5"))
        flow_net.deconv5.register_forward_hook(get_activation("deconv5"))
        flow_net.upfeat5.register_forward_hook(get_activation("upfeat5"))
        flow_net.conv4_0.register_forward_hook(get_activation("conv4_0"))
        flow_net.conv4_1.register_forward_hook(get_activation("conv4_1"))
        flow_net.conv4_2.register_forward_hook(get_activation("conv4_2"))
        flow_net.conv4_3.register_forward_hook(get_activation("conv4_3"))
        flow_net.conv4_4.register_forward_hook(get_activation("conv4_4"))
        flow_net.predict_flow4.register_forward_hook(get_activation("predict_flow4"))
        flow_net.deconv4.register_forward_hook(get_activation("deconv4"))
        flow_net.upfeat4.register_forward_hook(get_activation("upfeat4"))
        flow_net.conv3_0.register_forward_hook(get_activation("conv3_0"))
        flow_net.conv3_1.register_forward_hook(get_activation("conv3_1"))
        flow_net.conv3_2.register_forward_hook(get_activation("conv3_2"))
        flow_net.conv3_3.register_forward_hook(get_activation("conv3_3"))
        flow_net.conv3_4.register_forward_hook(get_activation("conv3_4"))
        flow_net.predict_flow3.register_forward_hook(get_activation("predict_flow3"))
        flow_net.deconv3.register_forward_hook(get_activation("deconv3"))
        flow_net.upfeat3.register_forward_hook(get_activation("upfeat3"))
        flow_net.conv2_0.register_forward_hook(get_activation("conv2_0"))
        flow_net.conv2_1.register_forward_hook(get_activation("conv2_1"))
        flow_net.conv2_2.register_forward_hook(get_activation("conv2_2"))
        flow_net.conv2_3.register_forward_hook(get_activation("conv2_3"))
        flow_net.conv2_4.register_forward_hook(get_activation("conv2_4"))
        flow_net.predict_flow2.register_forward_hook(get_activation("predict_flow2"))
        flow_net.dc_conv1.register_forward_hook(get_activation("dc_conv1"))
        flow_net.dc_conv2.register_forward_hook(get_activation("dc_conv2"))
        flow_net.dc_conv3.register_forward_hook(get_activation("dc_conv3"))
        flow_net.dc_conv4.register_forward_hook(get_activation("dc_conv4"))
        flow_net.dc_conv5.register_forward_hook(get_activation("dc_conv5"))
        flow_net.dc_conv6.register_forward_hook(get_activation("dc_conv6"))
        flow_net.dc_conv7.register_forward_hook(get_activation("dc_conv7"))
    elif "RAFT" in args.flownet or "FlowNetCFlex" in args.flownet:
        pass
    else:
        raise Exception("Flownet has to be specified for this experiment")


def get_feature_map_keys(args):
    if (
        args.flownet in ["FlowNetC", "FlowNetC_original", "FlowNetCFlexLarger_k3_reps3"]
        or "FlowNetCFlexLarger" in args.flownet
    ):
        return [
            "conv1a",
            "conv1b",
            "conv2a",
            "conv2b",
            "conv3a",
            "conv3b",
            "corr",
            "conv_redir",
            "conv3_1",
            "conv4",
            "conv4_1",
            "conv5",
            "conv5_1",
            "conv6",
            "conv6_1",
            "flow6",
            "upsampled_flow6_to_5",
            "deconv5",
            "flow5",
            "upsampled_flow5_to_4",
            "deconv4",
            "flow4",
            "upsampled_flow4_to_3",
            "deconv3",
            "flow3",
            "upsampled_flow3_to_2",
            "deconv2",
            "predict",
        ]
    elif "FlowNetCFlex" in args.flownet:
        return ["fmap1", "fmap2", "corr", "conv_redir", "predict_flow"]
    elif args.flownet in ["FlowNetS"]:
        return [
            "conv1",
            "conv2",
            "conv3",
            "conv3_1",
            "conv4",
            "conv4_1",
            "conv5",
            "conv5_1",
            "conv6",
            "conv6_1",
            "predict_flow6",
            "upsampled_flow6_to_5",
            "deconv5",
            "predict_flow5",
            "upsampled_flow5_to_4",
            "deconv4",
            "predict_flow4",
            "upsampled_flow4_to_3",
            "deconv3",
            "predict_flow3",
            "upsampled_flow3_to_2",
            "deconv2",
            "predict_flow2",
        ]
    elif args.flownet in ["PWCNet"]:
        return [
            "c11",
            "c21",
            "c12",
            "c22",
            "c13",
            "c23",
            "c14",
            "c24",
            "c15",
            "c25",
            "c16",
            "c26",
            "corr6",
            "conv6_0",
            "conv6_1",
            "conv6_2",
            "conv6_3",
            "conv6_4",
            "predict_flow6",
            "deconv6",
            "upfeat6",
            "corr5",
            "conv5_0",
            "conv5_1",
            "conv5_2",
            "conv5_3",
            "conv5_4",
            "predict_flow5",
            "deconv5",
            "upfeat5",
            "corr4",
            "conv4_0",
            "conv4_1",
            "conv4_2",
            "conv4_3",
            "conv4_4",
            "predict_flow4",
            "deconv4",
            "upfeat4",
            "corr3",
            "conv3_0",
            "conv3_1",
            "conv3_2",
            "conv3_3",
            "conv3_4",
            "predict_flow3",
            "deconv3",
            "upfeat3",
            "corr2",
            "conv2_0",
            "conv2_1",
            "conv2_2",
            "conv2_3",
            "conv2_4",
            "dc_conv1",
            "dc_conv2",
            "dc_conv3",
            "dc_conv4",
            "dc_conv5",
            "dc_conv6",
            "dc_conv7",
            "predict_flow2",
        ]
    elif "RAFT" in args.flownet:
        return (
            ["fmap1", "fmap2", "net", "inp"]
            + [f"corr_pyramid_{i}" for i in range(1 if "Corr1" in args.flownet else 4)]
            + [f"idx_corr_vol_{i}" for i in range(1 if "Iters1" in args.flownet else 12)]
            + [f"flow_pred_{i}" for i in range(1 if "Iters1" in args.flownet else 12)]
            + [f"net_{i}" for i in range(1 if "Iters1" in args.flownet else 12)]
            + [
                f"motion_features_{i}"
                for i in range(1 if "Iters1" in args.flownet else 12)
            ]
            + [f"cor1_{i}" for i in range(1 if "Iters1" in args.flownet else 12)]
            + [f"cor_{i}" for i in range(1 if "Iters1" in args.flownet else 12)]
            + [f"cor_flo_{i}" for i in range(1 if "Iters1" in args.flownet else 12)]
        )
    else:
        raise Exception("Flownet is currently not implemented")


activation = dict()


def get_activation(name):
    def hook(model, input, output):  # pylint: disable=unused-argument, redefined-builtin
        activation[name] = output.detach()

    return hook


def compute_feature_map(feature_map: np.ndarray, computation_type="mean") -> np.ndarray:
    assert computation_type in ["mean"]
    if computation_type == "mean":
        return np.mean(feature_map, axis=(-2, -1))


def get_feature_maps(args, feat_maps_to_numpy: bool = True) -> dict:
    if (
        args.flownet in ["FlowNetC", "FlowNetC_original", "FlowNetCFlexLarger_k3_reps3"]
        or "FlowNetCFlexLarger" in args.flownet
    ):
        list_of_feature_maps = [
            "conv_redir",
            "conv3_1",
            "conv4",
            "conv4_1",
            "conv5",
            "conv5_1",
            "conv6",
            "conv6_1",
            "flow6",
            "upsampled_flow6_to_5",
            "deconv5",
            "flow5",
            "upsampled_flow5_to_4",
            "deconv4",
            "flow4",
            "upsampled_flow4_to_3",
            "deconv3",
            "flow3",
            "upsampled_flow3_to_2",
            "deconv2",
            "predict",
        ]
    elif "FlowNetCFlex" in args.flownet:
        return dict()
    elif args.flownet in ["FlowNetS"]:
        list_of_feature_maps = [
            "conv1",
            "conv2",
            "conv3",
            "conv3_1",
            "conv4",
            "conv4_1",
            "conv5",
            "conv5_1",
            "conv6",
            "conv6_1",
            "predict_flow6",
            "upsampled_flow6_to_5",
            "deconv5",
            "predict_flow5",
            "upsampled_flow5_to_4",
            "deconv4",
            "predict_flow4",
            "upsampled_flow4_to_3",
            "deconv3",
            "predict_flow3",
            "upsampled_flow3_to_2",
            "deconv2",
            "predict_flow2",
        ]
    elif args.flownet in ["PWCNet"]:
        list_of_feature_maps = [
            "conv6_0",
            "conv6_1",
            "conv6_2",
            "conv6_3",
            "conv6_4",
            "predict_flow6",
            "deconv6",
            "upfeat6",
            "conv5_0",
            "conv5_1",
            "conv5_2",
            "conv5_3",
            "conv5_4",
            "predict_flow5",
            "deconv5",
            "upfeat5",
            "conv4_0",
            "conv4_1",
            "conv4_2",
            "conv4_3",
            "conv4_4",
            "predict_flow4",
            "deconv4",
            "upfeat4",
            "conv3_0",
            "conv3_1",
            "conv3_2",
            "conv3_3",
            "conv3_4",
            "predict_flow3",
            "deconv3",
            "upfeat3",
            "conv2_0",
            "conv2_1",
            "conv2_2",
            "conv2_3",
            "conv2_4",
            "predict_flow2",
            "dc_conv1",
            "dc_conv2",
            "dc_conv3",
            "dc_conv4",
            "dc_conv5",
            "dc_conv6",
            "dc_conv7",
        ]
    elif "RAFT" in args.flownet:
        return dict()

    if feat_maps_to_numpy:
        feature_maps = {
            fm: activation[fm].clone().cpu().numpy()[0, ...]
            for fm in list_of_feature_maps
        }
    else:
        feature_maps = {fm: activation[fm].clone() for fm in list_of_feature_maps}
    return feature_maps


def get_copied_feature_maps(feature_maps, copied_feat_maps, args):
    if (
        args.flownet in ["FlowNetC", "FlowNetC_original", "FlowNetCFlexLarger_k3_reps3"]
        or "FlowNetCFlexLarger" in args.flownet
    ):
        feature_maps["conv1a"] = copied_feat_maps[0]
        feature_maps["conv2a"] = copied_feat_maps[1]
        feature_maps["conv3a"] = copied_feat_maps[2]
        feature_maps["conv1b"] = copied_feat_maps[3]
        feature_maps["conv2b"] = copied_feat_maps[4]
        feature_maps["conv3b"] = copied_feat_maps[5]
        feature_maps["corr"] = copied_feat_maps[6]
    elif "FlowNetCFlex" in args.flownet:
        feature_maps["fmap1"] = copied_feat_maps[0]
        feature_maps["fmap2"] = copied_feat_maps[1]
        feature_maps["corr"] = copied_feat_maps[2]
        feature_maps["conv_redir"] = copied_feat_maps[3]
        feature_maps["predict_flow"] = copied_feat_maps[4]
    elif args.flownet in ["PWCNet"]:
        feature_maps["c11"] = copied_feat_maps[0]
        feature_maps["c21"] = copied_feat_maps[1]
        feature_maps["c12"] = copied_feat_maps[2]
        feature_maps["c22"] = copied_feat_maps[3]
        feature_maps["c13"] = copied_feat_maps[4]
        feature_maps["c23"] = copied_feat_maps[5]
        feature_maps["c14"] = copied_feat_maps[6]
        feature_maps["c24"] = copied_feat_maps[7]
        feature_maps["c15"] = copied_feat_maps[8]
        feature_maps["c25"] = copied_feat_maps[9]
        feature_maps["c16"] = copied_feat_maps[10]
        feature_maps["c26"] = copied_feat_maps[11]
        feature_maps["corr6"] = copied_feat_maps[12]
        feature_maps["corr5"] = copied_feat_maps[13]
        feature_maps["corr4"] = copied_feat_maps[14]
        feature_maps["corr3"] = copied_feat_maps[15]
        feature_maps["corr2"] = copied_feat_maps[16]
    elif "RAFT" in args.flownet:
        feature_maps["fmap1"] = copied_feat_maps[0]
        feature_maps["fmap2"] = copied_feat_maps[1]
        corr_pyramid_levels = 1 if "Corr1" in args.flownet else 4
        for i in range(corr_pyramid_levels):
            feature_maps[f"corr_pyramid_{i}"] = copied_feat_maps[i + 2]
        # corr_pyramid_levels += 1
        # feature_maps["spatial_corr"] = copied_feat_maps[corr_pyramid_levels + 1]
        feature_maps["net"] = copied_feat_maps[corr_pyramid_levels + 2]
        feature_maps["inp"] = copied_feat_maps[corr_pyramid_levels + 3]
        for i in range(1 if "Iters1" in args.flownet else 12):
            feature_maps[f"idx_corr_vol_{i}"] = copied_feat_maps[
                7 * i + corr_pyramid_levels + 4
            ]
            feature_maps[f"net_{i}"] = copied_feat_maps[7 * i + corr_pyramid_levels + 5]
            feature_maps[f"motion_features_{i}"] = copied_feat_maps[
                7 * i + corr_pyramid_levels + 6
            ]
            feature_maps[f"cor1_{i}"] = copied_feat_maps[7 * i + corr_pyramid_levels + 7]
            feature_maps[f"cor_{i}"] = copied_feat_maps[7 * i + corr_pyramid_levels + 8]
            feature_maps[f"cor_flo_{i}"] = copied_feat_maps[
                7 * i + corr_pyramid_levels + 9
            ]
            feature_maps[f"flow_pred_{i}"] = copied_feat_maps[
                7 * i + corr_pyramid_levels + 10
            ]
    return feature_maps


def predict_flow(
    flow_net,
    ref_past_img,  # pylint: disable=unused-argument
    tgt_img,
    ref_future_img,
    args,
    return_feat_maps=False,
    overwrite_feat_maps=None,
    feat_maps_to_numpy: bool = True,
):
    if return_feat_maps:
        if "RAFT" in args.flownet:
            _, flow_pred, copied_feat_maps = flow_net(
                image1=tgt_img * 255.0, image2=ref_future_img * 255.0, test_mode=True
            )
        elif "test" in flow_net.forward.__code__.co_varnames:
            if overwrite_feat_maps:
                flow_pred, copied_feat_maps = flow_net(
                    tgt_img,
                    ref_future_img,
                    overwrite_feat_maps=overwrite_feat_maps,
                    test=True,
                )
            else:
                flow_pred, copied_feat_maps = flow_net(tgt_img, ref_future_img, test=True)
        else:
            if overwrite_feat_maps:
                flow_pred, copied_feat_maps = flow_net(
                    tgt_img, ref_future_img, overwrite_feat_maps=overwrite_feat_maps
                )
            else:
                flow_pred, copied_feat_maps = flow_net(tgt_img, ref_future_img)
        if feat_maps_to_numpy:
            if len(copied_feat_maps) > 0:
                copied_feat_maps = [
                    fm.detach().cpu().numpy()[0, ...] for fm in copied_feat_maps
                ]
            feature_maps = get_feature_maps(args, True)
            feature_maps = get_copied_feature_maps(feature_maps, copied_feat_maps, args)
        else:
            if len(copied_feat_maps) > 0:
                copied_feat_maps = [fm.clone() for fm in copied_feat_maps]
            feature_maps = get_feature_maps(args, False)
            feature_maps = get_copied_feature_maps(feature_maps, copied_feat_maps, args)
        return flow_pred, feature_maps
    else:
        if "RAFT" in args.flownet:
            _, flow_pred = flow_net(
                image1=tgt_img * 255.0, image2=ref_future_img * 255.0, test_mode=True
            )
        elif "test" in flow_net.forward.__code__.co_varnames:
            flow_pred = flow_net(tgt_img, ref_future_img, test=True)
        else:
            flow_pred = flow_net(tgt_img, ref_future_img)
        return flow_pred


if __name__ == "__main__":
    from argparse import Namespace

    from tqdm import tqdm

    pbar = tqdm(get_flownet_choices())
    for flownet in pbar:
        pbar.set_description(f"Flownet: {flownet}")
        args = Namespace()
        args.flownet = flownet
        try:
            _ = fetch_model(args)
        except Exception as e:
            print(
                "RAFT model needs to be converted. See https://github.com/pytorch/pytorch/issues/48915 for more information."
            )
