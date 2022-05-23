# Taken from https://github.com/ClementPinard/SfmLearner-Pytorch/


import argparse

import numpy as np
from joblib import Parallel, delayed
from path import Path
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar="DIR", help="path to original dataset")
parser.add_argument(
    "--dataset-format", type=str, required=True, choices=["kitti", "cityscapes"]
)
parser.add_argument(
    "--static-frames",
    default=None,
    help="list of imgs to discard for being static, if not set will discard them based on speed \
                    (careful, on KITTI some frames have incorrect speed)",
)
parser.add_argument(
    "--with-gt",
    action="store_true",
    help="If available (e.g. with KITTI), will store ground truth along with images, for validation",
)
parser.add_argument("--dump-root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--height", type=int, default=128, help="image height")
parser.add_argument("--width", type=int, default=416, help="image width")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

args = parser.parse_args()


def dump_example(scene):
    scene_list = data_loader.collect_scenes(scene)
    for scene_data in scene_list:
        dump_dir = args.dump_root / scene_data["rel_path"]
        dump_dir.makedirs_p()
        intrinsics = scene_data["intrinsics"]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        dump_cam_file = dump_dir / "cam.txt"
        with open(dump_cam_file, "w", encoding="utf-8") as f:
            f.write(f"{fx:f},0.,{cx:f},0.,{fy:f},{cy:f},0.,0.,1.")

        for sample in data_loader.get_scene_imgs(scene_data):
            assert len(sample) >= 2
            img, frame_nb = sample[0], sample[1]
            dump_img_file = dump_dir / f"{frame_nb}.jpg"
            Image.fromarray(img).save(dump_img_file)
            if len(sample) == 3:
                dump_depth_file = dump_dir / f"{frame_nb}.npy"
                np.save(dump_depth_file, sample[2])

        if len(dump_dir.files("*.jpg")) < 3:
            dump_dir.rmtree()


def main():
    args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader  # pylint: disable=global-variable-undefined

    if args.dataset_format == "kitti":
        from setup.kitti_raw_loader import KittiRawLoader

        data_loader = KittiRawLoader(
            args.dataset_dir,
            static_frames_file=args.static_frames,
            img_height=args.height,
            img_width=args.width,
            get_gt=args.with_gt,
        )

    if args.dataset_format == "cityscapes":
        raise NotImplementedError
        # from cityscapes_loader import cityscapes_loader

        # data_loader = cityscapes_loader(
        #     args.dataset_dir, img_height=args.height, img_width=args.width
        # )

    print("Retrieving frames")
    Parallel(n_jobs=args.num_threads)(
        delayed(dump_example)(scene) for scene in tqdm(data_loader.scenes)
    )
    # Split into train/val
    print("Generating train val lists")
    np.random.seed(8964)
    subfolders = args.dump_root.dirs()
    with open(args.dump_root / "train.txt", "w", encoding="utf-8") as tf:
        with open(args.dump_root / "val.txt", "w", encoding="utf-8") as vf:
            for s in tqdm(subfolders):
                # gt_missing = False
                # for f in os.listdir(s):
                #    if 'jpg' in f:
                #        if not os.path.exists(f.replace('jpg', 'npy')):
                #            gt_missing = True
                #            print('To val since no gt exist ' + s)
                #            break
                if np.random.random() < 0.1:
                    vf.write(f"{s.name}\n")
                else:
                    tf.write(f"{s.name}\n")
                    # remove useless groundtruth data for training comment if you don't want to erase it
                    # for gt_file in s.files('*.npy'):
                    #    gt_file.remove_p()


if __name__ == "__main__":
    main()
