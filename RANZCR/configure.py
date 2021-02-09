import os


class CFG:
    use_amp = False
    debug = False

    train_df_fp = "data/train_folds.csv"
    test_df_fp = "data/sample_submission.csv"
    train_img_dir = "data/train"
    test_img_dir = "data/test"
    do_train = False
    do_predict = False
    checkpoint = "outputs/checkpoints/resnet18/epoch=00-val_auc=0.0000-v0.ckpt"
    num_workers = 4
    model_name = "resnet18"
    image_size = 512
    fold = 1
    gpus = [0]
    batch_size_by_devices = {
        "tpu": 10,
        "gpu": 128,
        "cpu": 4,
    }
    batch_size = batch_size_by_devices["gpu"]
    seed = 1710

    target_cols = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]

    target_size = len(target_cols)
    fold = 0
    output_dir = os.path.join("outputs", "checkpoints", model_name)
    submission_dir = os.path.join("outputs", "results", model_name)
    color = "\033[32m"
