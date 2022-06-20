def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'TinyVirat':
        cfg.data_folder = 'datasets/TinyVIRAT-v2-drop2/videos'
        cfg.train_annotations = 'datasets/TinyVIRAT-v2-drop2/tiny_train_v2.json'
        cfg.val_annotations = 'datasets/TinyVIRAT-v2-drop2/tiny_val_v2.json'
        cfg.test_annotations = 'datasets/TinyVIRAT-v2-drop2/tiny_test_v2_public.json'
        cfg.class_map = 'datasets/TinyVIRAT-v2-drop2/class_map.json'
        cfg.num_classes = 26
    elif dataset == 'ucf101':
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
        cfg.data_folder = r"D:\DATASET\myaction\ConstructionSafety"
        cfg.train_split = r"D:\DATASET\myaction\train.txt"
        cfg.val_split = r"D:\DATASET\myaction\test.txt"
        #cfg.test_split = r"D:\DATASET\myaction\test_class.txt"
        cfg.test_split = r"D:\DATASET\myaction\test.txt"
        cfg.dataset_path = r"D:\DATASET\myaction"
        cfg.num_classes = 7
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    return cfg

