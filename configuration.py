def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == "myaction":
        cfg.data_folder = r"D:\DATASET\dataset\myaction"              # location to myaction
        cfg.train_split = r"D:\DATASET\dataset\train.txt"             # location to train split
        cfg.val_split = r"D:\DATASET\dataset\test.txt"
        cfg.test_split = r"D:\DATASET\dataset\test.txt"               # location to test split
        cfg.dataset_path = r"D:\DATASET\dataset"                      # location to dataset
        cfg.num_classes = 7
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    return cfg

