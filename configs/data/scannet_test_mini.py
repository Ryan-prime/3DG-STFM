from configs.data.base import cfg

TEST_BASE_PATH = "data/scannet_mini/index"

cfg.DATASET.TEST_DATA_SOURCE = "ScanNet"
cfg.DATASET.TEST_DATA_ROOT = "data/scannet_mini/train"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_data/train"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/scene_data/train_list/scannet_all.txt"
cfg.DATASET.TEST_INTRINSIC_PATH = f"{TEST_BASE_PATH}/intrinsics.npz"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.4