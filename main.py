import pandas as pd
import pytorch_lightning as pl

from RANZCR.configure import CFG
from RANZCR.data import RANZCRDataModule
from RANZCR.model import RANZCRModel
from RANZCR.utils import print_signature, seed_everything

if __name__ == "__main__":

    seed_everything(seed=CFG.seed)
    print_signature()
    print("*** GLOBAL CONFIGURATION ***")
    for k, v in vars(CFG).items():
        print(k, ": ", v)
    print("****************************")
    train_df = pd.read_csv(CFG.train_df_fp)
    test_df = pd.read_csv(CFG.test_df_fp)

    print(train_df.head())
    print(test_df.head())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=CFG.output_dir + "/{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        #     save_top_k=1,
    )
    trainer = pl.Trainer(
        gpus=CFG.gpus,
        precision=16,
        max_epochs=10,
        num_sanity_val_steps=1 if CFG.debug else 0,
        checkpoint_callback=checkpoint_callback,
        #     val_check_interval=0.5,  # check validation twice per epoch
    )
    model = RANZCRModel(test_df=test_df)
    data_module = RANZCRDataModule(
        train_df=train_df,
        test_df=test_df,
        train_img_dir=CFG.train_img_dir,
        test_img_dir=CFG.test_img_dir,
        fold=CFG.fold,
    )
    if CFG.do_train:
        trainer.fit(model, data_module)
    if CFG.do_predict:
        trainer.test(model, datamodule=data_module, ckpt_path=CFG.checkpoint)
