def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == "myaction":
        cfg.data_folder = r"D:\DATASET\myaction\ConstructionSafety"    # location to CMA dataset
        cfg.train_split = r"D:\DATASET\myaction\train.txt"             # location to train split
        cfg.val_split = r"D:\DATASET\myaction\test.txt"
        cfg.test_split = r"D:\DATASET\myaction\test.txt"               # location to test split
        cfg.dataset_path = r"D:\DATASET\myaction"
        cfg.num_classes = 7
    cfg.saved_models_dir = './results/saved_models'
    cfg.tf_logs_dir = './results/logs'
    return cfg

