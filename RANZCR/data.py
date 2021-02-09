import os

import albumentations
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from RANZCR.configure import CFG


class RANZCRDataset(Dataset):
    def __init__(self, df, img_dir, mode, image_size=CFG.image_size, target_cols=CFG.target_cols):

        self.df = self._get_df(df.copy(), img_dir)
        self.labels = self.df[target_cols].values
        self.mode = mode
        self.image_size = image_size

        self._setup_transform()
        if mode == "train":
            self.transform = self.transform_train
        elif mode == "val":
            self.transform = self.transform_val
        elif mode == "test":
            self.transform = self.transform_test
        else:
            raise ValueError(f"Invalid mode {mode}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img = cv2.imread(data["file_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        label = torch.tensor(self.labels[idx]).float()
        if self.mode == "test":
            return img.clone().detach()
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        return torch.tensor(img).float(), label

    def _setup_transform(self):
        self.transform_train = albumentations.Compose(
            [
                albumentations.RandomResizedCrop(self.image_size, self.image_size, scale=(0.9, 1), p=1),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ShiftScaleRotate(p=0.5),
                albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7
                ),
                albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
                albumentations.OneOf(
                    [
                        albumentations.OpticalDistortion(distort_limit=1.0),
                        albumentations.GridDistortion(num_steps=5, distort_limit=1.0),
                        albumentations.ElasticTransform(alpha=3),
                    ],
                    p=0.2,
                ),
                albumentations.OneOf(
                    [
                        albumentations.GaussNoise(var_limit=[10, 50]),
                        albumentations.GaussianBlur(),
                        albumentations.MotionBlur(),
                        albumentations.MedianBlur(),
                    ],
                    p=0.2,
                ),
                albumentations.Resize(self.image_size, self.image_size),
                albumentations.OneOf(
                    [
                        albumentations.JpegCompression(),
                        albumentations.Downscale(scale_min=0.1, scale_max=0.15),
                    ],
                    p=0.2,
                ),
                albumentations.IAAPiecewiseAffine(p=0.2),
                albumentations.IAASharpen(p=0.2),
                albumentations.Cutout(
                    max_h_size=int(self.image_size * 0.1),
                    max_w_size=int(self.image_size * 0.1),
                    num_holes=5,
                    p=0.5,
                ),
                albumentations.Normalize(),
            ]
        )

        self.transform_val = albumentations.Compose(
            [
                albumentations.Resize(self.image_size, self.image_size),
                albumentations.Normalize(),
            ]
        )
        self.transform_test = albumentations.Compose(
            [
                albumentations.Resize(self.image_size, self.image_size),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    def _get_df(self, df, img_dir):
        df["file_path"] = df["StudyInstanceUID"].apply(lambda id_: os.path.join(img_dir, id_ + ".jpg"))
        df = df.reset_index(drop=True)
        return df


class RANZCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        test_df,
        fold,
        train_img_dir=CFG.train_img_dir,
        test_img_dir=CFG.test_img_dir,
        batch_size=CFG.batch_size,
        image_size=CFG.image_size,
    ):
        super().__init__()
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

        #         split train-val
        self.val_df = self.train_df[self.train_df["fold"] == fold]
        self.train_df = self.train_df[self.train_df["fold"] != fold]

        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.batch_size = batch_size
        self.image_size = image_size

        # NOTE: for debugging only
        # self.train_df = self.train_df.iloc[:100]
        # self.val_df = self.val_df.iloc[:100]
        # self.test_df = self.test_df.iloc[:100]

    def setup(self, stage=None):

        self.RANZCR_train = RANZCRDataset(
            df=self.train_df,
            img_dir=self.train_img_dir,
            mode="train",
            image_size=self.image_size,
        )
        self.RANZCR_val = RANZCRDataset(
            df=self.val_df,
            img_dir=self.train_img_dir,
            mode="val",
            image_size=self.image_size,
        )
        self.RANZCR_test = RANZCRDataset(
            df=self.test_df,
            img_dir=self.test_img_dir,
            mode="test",
            image_size=self.image_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.RANZCR_train,
            batch_size=self.batch_size,
            num_workers=CFG.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.RANZCR_val,
            batch_size=self.batch_size,
            num_workers=CFG.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.RANZCR_test,
            batch_size=self.batch_size,
            num_workers=CFG.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=False,
        )
