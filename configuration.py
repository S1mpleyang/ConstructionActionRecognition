def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'ucf101':
        cfg.data_folder = r"D:\DATASET\ucf101\UCF-101"
        cfg.train_split = r"D:\DATASET\ucf101\ucf101_train_split_1_videos.txt"
        cfg.val_split = r"D:\DATASET\ucf101\ucf101_val_split_1_videos.txt"
        cfg.test_split = r"D:\DATASET\ucf101\ucf101_val_split_1_videos.txt"
        cfg.num_classes = 101
    elif dataset == 'hmdb51':
        cfg.data_folder = r"D:\DATASET\HMDB51\hmdb51_unzip"
        cfg.train_split = r"D:\DATASET\HMDB51\hmdb51_train_split1.txt"
        cfg.val_split = r"D:\DATASET\HMDB51\hmdb51_val_split1.txt"
        cfg.test_split = r"D:\DATASET\HMDB51\hmdb51_test_split1.txt"
        cfg.dataset_path = r"D:\DATASET\HMDB51"
        cfg.num_classes = 51
    elif dataset == "myaction":
        cfg.data_folder = r"D:\DATASET\myaction\ConstructionSafety"    # location to CMA dataset
        cfg.train_split = r"D:\DATASET\myaction\train.txt"             # location to train split
        cfg.val_split = r"D:\DATASET\myaction\test.txt"
        cfg.test_split = r"D:\DATASET\myaction\test.txt"               # location to test split
        cfg.dataset_path = r"D:\DATASET\myaction"
        cfg.num_classes = 7
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    return cfg

