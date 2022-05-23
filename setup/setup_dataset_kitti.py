# Taken from https://github.com/alexklwong/stereopagnosia

import glob
import os

from dataset_utils import data_utils

# Training and testing file output directories
TRAIN_REFS_DIRPATH = os.path.join("datasets", "KITTI", "global_attacks", "training")
TEST_REFS_DIRPATH = os.path.join("datasets", "KITTI", "global_attacks", "testing")

# KITTI 2012 dataset (stereo flow)
STEREO_FLOW_ROOT_DIRPATH = os.path.join("datasets", "KITTI", "2012", "training")

STEREO_FLOW_IMAGE0_DIRPATH = os.path.join(STEREO_FLOW_ROOT_DIRPATH, "colored_0")
STEREO_FLOW_IMAGE1_DIRPATH = os.path.join(STEREO_FLOW_ROOT_DIRPATH, "colored_1")
STEREO_FLOW_DISPARITY_DIRPATH = os.path.join(STEREO_FLOW_ROOT_DIRPATH, "disp_occ")
STEREO_FLOW_FLOW_DIRPATH = os.path.join(STEREO_FLOW_ROOT_DIRPATH, "flow_occ")

STEREO_FLOW_ALL_IMAGE0_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_all_image0.txt"
)
STEREO_FLOW_ALL_IMAGE1_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_all_image1.txt"
)
STEREO_FLOW_ALL_DISPARITY_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_all_disparity.txt"
)
STEREO_FLOW_ALL_FLOW_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_all_flow.txt"
)

STEREO_FLOW_TRAIN_IMAGE0_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_train_image0.txt"
)
STEREO_FLOW_TRAIN_IMAGE1_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_train_image1.txt"
)
STEREO_FLOW_TRAIN_DISPARITY_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_train_disparity.txt"
)
STEREO_FLOW_TRAIN_FLOW_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_flow_train_flow.txt"
)

STEREO_FLOW_TEST_IMAGE0_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_flow_test_image0.txt"
)
STEREO_FLOW_TEST_IMAGE1_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_flow_test_image1.txt"
)
STEREO_FLOW_TEST_DISPARITY_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_flow_test_disparity.txt"
)
STEREO_FLOW_TEST_FLOW_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_flow_test_flow.txt"
)

# KITTI 2015 dataset (scene flow)
SCENE_FLOW_ROOT_DIRPATH = os.path.join("datasets", "KITTI", "2015", "training")

SCENE_FLOW_IMAGE0_DIRPATH = os.path.join(SCENE_FLOW_ROOT_DIRPATH, "image_2")
SCENE_FLOW_IMAGE1_DIRPATH = os.path.join(SCENE_FLOW_ROOT_DIRPATH, "image_3")
SCENE_FLOW_DISPARITY_DIRPATH = os.path.join(SCENE_FLOW_ROOT_DIRPATH, "disp_occ_0")
SCENE_FLOW_FLOW_DIRPATH = os.path.join(SCENE_FLOW_ROOT_DIRPATH, "flow_occ")

SCENE_FLOW_ALL_IMAGE0_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_all_image0.txt"
)
SCENE_FLOW_ALL_IMAGE1_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_all_image1.txt"
)
SCENE_FLOW_ALL_DISPARITY_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_all_disparity.txt"
)
SCENE_FLOW_ALL_FLOW_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_all_flow.txt"
)

SCENE_FLOW_TRAIN_IMAGE0_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_train_image0.txt"
)
SCENE_FLOW_TRAIN_IMAGE1_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_train_image1.txt"
)
SCENE_FLOW_TRAIN_DISPARITY_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_train_disparity.txt"
)
SCENE_FLOW_TRAIN_FLOW_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_scene_flow_train_flow.txt"
)

SCENE_FLOW_TEST_IMAGE0_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_scene_flow_test_image0.txt"
)
SCENE_FLOW_TEST_IMAGE1_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_scene_flow_test_image1.txt"
)
SCENE_FLOW_TEST_DISPARITY_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_scene_flow_test_disparity.txt"
)
SCENE_FLOW_TEST_FLOW_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_scene_flow_test_flow.txt"
)

# KITTI 2012 (stereo flow) mixed with KITTI 2015 (scene flow) dataset
STEREO_SCENE_FLOW_ALL_IMAGE0_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_all_image0.txt"
)
STEREO_SCENE_FLOW_ALL_IMAGE1_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_all_image1.txt"
)
STEREO_SCENE_FLOW_ALL_DISPARITY_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_all_disparity.txt"
)
STEREO_SCENE_FLOW_ALL_FLOW_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_all_flow.txt"
)

STEREO_SCENE_FLOW_TRAIN_IMAGE0_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_train_image0.txt"
)
STEREO_SCENE_FLOW_TRAIN_IMAGE1_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_train_image1.txt"
)
STEREO_SCENE_FLOW_TRAIN_DISPARITY_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_train_disparity.txt"
)
STEREO_SCENE_FLOW_TRAIN_FLOW_FILEPATH = os.path.join(
    TRAIN_REFS_DIRPATH, "kitti_stereo_scene_flow_train_flow.txt"
)

STEREO_SCENE_FLOW_TEST_IMAGE0_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_scene_flow_test_image0.txt"
)
STEREO_SCENE_FLOW_TEST_IMAGE1_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_scene_flow_test_image1.txt"
)
STEREO_SCENE_FLOW_TEST_DISPARITY_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_scene_flow_test_disparity.txt"
)
STEREO_SCENE_FLOW_TEST_FLOW_FILEPATH = os.path.join(
    TEST_REFS_DIRPATH, "kitti_stereo_scene_flow_test_flow.txt"
)


# Create training and testing output directories
for dirpath in [TRAIN_REFS_DIRPATH, TEST_REFS_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


"""
Process KITTI 2012 dataset (stereo flow) file paths
"""
stereo_flow_image0_paths = sorted(
    glob.glob(os.path.join(STEREO_FLOW_IMAGE0_DIRPATH, "*_10.png"))
)
stereo_flow_image1_paths = sorted(
    glob.glob(os.path.join(STEREO_FLOW_IMAGE1_DIRPATH, "*_10.png"))
)
stereo_flow_disparity_paths = sorted(
    glob.glob(os.path.join(STEREO_FLOW_DISPARITY_DIRPATH, "*_10.png"))
)
stereo_flow_flow_paths = sorted(
    glob.glob(os.path.join(STEREO_FLOW_FLOW_DIRPATH, "*_10.png"))
)

assert len(stereo_flow_image0_paths) == len(stereo_flow_image1_paths)
assert len(stereo_flow_image0_paths) == len(stereo_flow_disparity_paths)
assert len(stereo_flow_image0_paths) == len(stereo_flow_flow_paths)

# Split KITTI 2012 dataset into training and test sets
stereo_flow_train_image0_paths = stereo_flow_image0_paths[0:160]
stereo_flow_train_image1_paths = stereo_flow_image1_paths[0:160]
stereo_flow_train_disparity_paths = stereo_flow_disparity_paths[0:160]
stereo_flow_train_flow_paths = stereo_flow_flow_paths[0:160]

stereo_flow_test_image0_paths = stereo_flow_image0_paths[160:]
stereo_flow_test_image1_paths = stereo_flow_image1_paths[160:]
stereo_flow_test_disparity_paths = stereo_flow_disparity_paths[160:]
stereo_flow_test_flow_paths = stereo_flow_flow_paths[160:]

# Write all paths to disk
print(
    "Storing all {} stereo flow left stereo images file paths into: {}".format(
        len(stereo_flow_image0_paths), STEREO_FLOW_ALL_IMAGE0_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_ALL_IMAGE0_FILEPATH, stereo_flow_image0_paths)

print(
    "Storing all {} stereo flow right stereo images file paths into: {}".format(
        len(stereo_flow_image1_paths), STEREO_FLOW_ALL_IMAGE1_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_ALL_IMAGE1_FILEPATH, stereo_flow_image1_paths)

print(
    "Storing all {} stereo flow ground truth disparity file paths into: {}".format(
        len(stereo_flow_disparity_paths), STEREO_FLOW_ALL_DISPARITY_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_ALL_DISPARITY_FILEPATH, stereo_flow_disparity_paths)

print(
    "Storing all {} stereo flow ground truth flow file paths into: {}".format(
        len(stereo_flow_flow_paths), STEREO_FLOW_ALL_FLOW_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_ALL_FLOW_FILEPATH, stereo_flow_flow_paths)

# Write training paths to disk
print(
    "Storing {} training stereo flow left stereo images file paths into: {}".format(
        len(stereo_flow_train_image0_paths), STEREO_FLOW_TRAIN_IMAGE0_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_TRAIN_IMAGE0_FILEPATH, stereo_flow_train_image0_paths)

print(
    "Storing {} training stereo flow right stereo images file paths into: {}".format(
        len(stereo_flow_train_image1_paths), STEREO_FLOW_TRAIN_IMAGE1_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_TRAIN_IMAGE1_FILEPATH, stereo_flow_train_image1_paths)

print(
    "Storing {} training stereo flow ground truth disparity file paths into: {}".format(
        len(stereo_flow_train_disparity_paths), STEREO_FLOW_TRAIN_DISPARITY_FILEPATH
    )
)
data_utils.write_paths(
    STEREO_FLOW_TRAIN_DISPARITY_FILEPATH, stereo_flow_train_disparity_paths
)

print(
    "Storing {} training stereo flow ground truth flow file paths into: {}".format(
        len(stereo_flow_train_flow_paths), STEREO_FLOW_TRAIN_FLOW_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_TRAIN_FLOW_FILEPATH, stereo_flow_train_flow_paths)

# Write testing paths to disk
print(
    "Storing {} testing stereo flow left stereo images file paths into: {}".format(
        len(stereo_flow_test_image0_paths), STEREO_FLOW_TEST_IMAGE0_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_TEST_IMAGE0_FILEPATH, stereo_flow_test_image0_paths)

print(
    "Storing {} testing stereo flow right stereo images file paths into: {}".format(
        len(stereo_flow_test_image1_paths), STEREO_FLOW_TEST_IMAGE1_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_TEST_IMAGE1_FILEPATH, stereo_flow_test_image1_paths)

print(
    "Storing {} testing stereo flow ground truth disparity file paths into: {}".format(
        len(stereo_flow_test_disparity_paths), STEREO_FLOW_TEST_DISPARITY_FILEPATH
    )
)
data_utils.write_paths(
    STEREO_FLOW_TEST_DISPARITY_FILEPATH, stereo_flow_test_disparity_paths
)

print(
    "Storing {} testing stereo flow ground truth flow file paths into: {}".format(
        len(stereo_flow_test_flow_paths), STEREO_FLOW_TEST_FLOW_FILEPATH
    )
)
data_utils.write_paths(STEREO_FLOW_TEST_FLOW_FILEPATH, stereo_flow_test_flow_paths)


"""
Process KITTI 2015 dataset (scene flow) file paths
"""
scene_flow_image0_paths = sorted(
    glob.glob(os.path.join(SCENE_FLOW_IMAGE0_DIRPATH, "*_10.png"))
)
scene_flow_image1_paths = sorted(
    glob.glob(os.path.join(SCENE_FLOW_IMAGE1_DIRPATH, "*_10.png"))
)
scene_flow_disparity_paths = sorted(
    glob.glob(os.path.join(SCENE_FLOW_DISPARITY_DIRPATH, "*_10.png"))
)
scene_flow_flow_paths = sorted(
    glob.glob(os.path.join(SCENE_FLOW_FLOW_DIRPATH, "*_10.png"))
)

assert len(scene_flow_image0_paths) == len(scene_flow_image1_paths)
assert len(scene_flow_image0_paths) == len(scene_flow_disparity_paths)
assert len(scene_flow_image0_paths) == len(scene_flow_flow_paths)

# Split KITTI 2012 dataset into training and test sets
scene_flow_train_image0_paths = scene_flow_image0_paths[0:160]
scene_flow_train_image1_paths = scene_flow_image1_paths[0:160]
scene_flow_train_disparity_paths = scene_flow_disparity_paths[0:160]
scene_flow_train_flow_paths = scene_flow_flow_paths[0:160]

scene_flow_test_image0_paths = scene_flow_image0_paths[160:]
scene_flow_test_image1_paths = scene_flow_image1_paths[160:]
scene_flow_test_disparity_paths = scene_flow_disparity_paths[160:]
scene_flow_test_flow_paths = scene_flow_flow_paths[160:]

# Write all paths to disk
print(
    "Storing all {} scene flow left stereo images file paths into: {}".format(
        len(scene_flow_image0_paths), SCENE_FLOW_ALL_IMAGE0_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_ALL_IMAGE0_FILEPATH, scene_flow_image0_paths)

print(
    "Storing all {} scene flow right stereo images file paths into: {}".format(
        len(scene_flow_image1_paths), SCENE_FLOW_ALL_IMAGE1_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_ALL_IMAGE1_FILEPATH, scene_flow_image1_paths)

print(
    "Storing all {} scene flow ground truth disparity file paths into: {}".format(
        len(scene_flow_disparity_paths), SCENE_FLOW_ALL_DISPARITY_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_ALL_DISPARITY_FILEPATH, scene_flow_disparity_paths)

print(
    "Storing all {} scene flow ground truth flow file paths into: {}".format(
        len(scene_flow_flow_paths), SCENE_FLOW_ALL_FLOW_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_ALL_FLOW_FILEPATH, scene_flow_flow_paths)

# Write training paths to disk
print(
    "Storing {} training scene flow left stereo images file paths into: {}".format(
        len(scene_flow_train_image0_paths), SCENE_FLOW_TRAIN_IMAGE0_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_TRAIN_IMAGE0_FILEPATH, scene_flow_train_image0_paths)

print(
    "Storing {} training scene flow right stereo images file paths into: {}".format(
        len(scene_flow_train_image1_paths), SCENE_FLOW_TRAIN_IMAGE1_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_TRAIN_IMAGE1_FILEPATH, scene_flow_train_image1_paths)

print(
    "Storing {} training scene flow ground truth disparity file paths into: {}".format(
        len(scene_flow_train_disparity_paths), SCENE_FLOW_TRAIN_DISPARITY_FILEPATH
    )
)
data_utils.write_paths(
    SCENE_FLOW_TRAIN_DISPARITY_FILEPATH, scene_flow_train_disparity_paths
)

print(
    "Storing {} training scene flow ground truth flow file paths into: {}".format(
        len(scene_flow_train_flow_paths), SCENE_FLOW_TRAIN_FLOW_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_TRAIN_FLOW_FILEPATH, scene_flow_train_flow_paths)

# Write testing paths to disk
print(
    "Storing {} testing scene flow left stereo images file paths into: {}".format(
        len(scene_flow_test_image0_paths), SCENE_FLOW_TEST_IMAGE0_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_TEST_IMAGE0_FILEPATH, scene_flow_test_image0_paths)

print(
    "Storing {} testing scene flow right stereo images file paths into: {}".format(
        len(scene_flow_test_image1_paths), SCENE_FLOW_TEST_IMAGE1_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_TEST_IMAGE1_FILEPATH, scene_flow_test_image1_paths)

print(
    "Storing {} testing scene flow ground truth disparity file paths into: {}".format(
        len(scene_flow_test_disparity_paths), SCENE_FLOW_TEST_DISPARITY_FILEPATH
    )
)
data_utils.write_paths(
    SCENE_FLOW_TEST_DISPARITY_FILEPATH, scene_flow_test_disparity_paths
)

print(
    "Storing {} testing scene flow ground truth flow file paths into: {}".format(
        len(scene_flow_test_flow_paths), SCENE_FLOW_TEST_FLOW_FILEPATH
    )
)
data_utils.write_paths(SCENE_FLOW_TEST_FLOW_FILEPATH, scene_flow_test_flow_paths)


"""
Combine KITTI 2012 with KITTI 2015 dataset
"""
# Write all paths to disk
print(
    "Storing all {} stereo flow + scene flow left stereo images file paths into: {}".format(
        len(stereo_flow_image0_paths + scene_flow_image0_paths),
        STEREO_SCENE_FLOW_ALL_IMAGE0_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_ALL_IMAGE0_FILEPATH,
    stereo_flow_image0_paths + scene_flow_image0_paths,
)

print(
    "Storing all {} stereo flow + scene flow right stereo images file paths into: {}".format(
        len(stereo_flow_image1_paths + scene_flow_image1_paths),
        STEREO_SCENE_FLOW_ALL_IMAGE1_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_ALL_IMAGE1_FILEPATH,
    stereo_flow_image1_paths + scene_flow_image1_paths,
)

print(
    "Storing all {} stereo flow + scene flow ground truth disparity file paths into: {}".format(
        len(stereo_flow_disparity_paths + scene_flow_disparity_paths),
        STEREO_SCENE_FLOW_ALL_DISPARITY_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_ALL_DISPARITY_FILEPATH,
    stereo_flow_disparity_paths + scene_flow_disparity_paths,
)

print(
    "Storing all {} stereo flow + scene flow ground truth flow file paths into: {}".format(
        len(stereo_flow_flow_paths + scene_flow_flow_paths),
        STEREO_SCENE_FLOW_ALL_FLOW_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_ALL_FLOW_FILEPATH, stereo_flow_flow_paths + scene_flow_flow_paths
)

# Write training paths to disk
print(
    "Storing {} training stereo flow + scene flow left stereo images file paths into: {}".format(
        len(stereo_flow_train_image0_paths + scene_flow_train_image0_paths),
        STEREO_SCENE_FLOW_TRAIN_IMAGE0_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TRAIN_IMAGE0_FILEPATH,
    stereo_flow_train_image0_paths + scene_flow_train_image0_paths,
)

print(
    "Storing {} training stereo flow + scene flow right stereo images file paths into: {}".format(
        len(stereo_flow_train_image1_paths + scene_flow_train_image1_paths),
        STEREO_SCENE_FLOW_TRAIN_IMAGE1_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TRAIN_IMAGE1_FILEPATH,
    stereo_flow_train_image1_paths + scene_flow_train_image1_paths,
)

print(
    "Storing {} training stereo flow + scene flow ground truth disparity file paths into: {}".format(
        len(stereo_flow_train_disparity_paths + scene_flow_train_disparity_paths),
        STEREO_SCENE_FLOW_TRAIN_DISPARITY_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TRAIN_DISPARITY_FILEPATH,
    stereo_flow_train_disparity_paths + scene_flow_train_disparity_paths,
)

print(
    "Storing {} training stereo flow + scene flow ground truth flow file paths into: {}".format(
        len(stereo_flow_train_flow_paths + scene_flow_train_flow_paths),
        STEREO_SCENE_FLOW_TRAIN_FLOW_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TRAIN_FLOW_FILEPATH,
    stereo_flow_train_flow_paths + scene_flow_train_flow_paths,
)

# Write testing paths to disk
print(
    "Storing {} testing stereo flow + scene flow left stereo images file paths into: {}".format(
        len(stereo_flow_test_image0_paths + scene_flow_test_image0_paths),
        STEREO_SCENE_FLOW_TEST_IMAGE0_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TEST_IMAGE0_FILEPATH,
    stereo_flow_test_image0_paths + scene_flow_test_image0_paths,
)

print(
    "Storing {} testing stereo flow + scene flow right stereo images file paths into: {}".format(
        len(stereo_flow_test_image1_paths + scene_flow_test_image1_paths),
        STEREO_SCENE_FLOW_TEST_IMAGE1_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TEST_IMAGE1_FILEPATH,
    stereo_flow_test_image1_paths + scene_flow_test_image1_paths,
)

print(
    "Storing {} testing stereo flow + scene flow ground truth disparity file paths into: {}".format(
        len(stereo_flow_test_disparity_paths + scene_flow_test_disparity_paths),
        STEREO_SCENE_FLOW_TEST_DISPARITY_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TEST_DISPARITY_FILEPATH,
    stereo_flow_test_disparity_paths + scene_flow_test_disparity_paths,
)

print(
    "Storing {} testing stereo flow + scene flow ground truth flow file paths into: {}".format(
        len(stereo_flow_test_flow_paths + scene_flow_test_flow_paths),
        STEREO_SCENE_FLOW_TEST_FLOW_FILEPATH,
    )
)
data_utils.write_paths(
    STEREO_SCENE_FLOW_TEST_FLOW_FILEPATH,
    stereo_flow_test_flow_paths + scene_flow_test_flow_paths,
)
