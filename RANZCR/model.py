import os

import matplotlib as plt
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, roc_auc_score, roc_curve

from RANZCR.configure import CFG


class RANZCRModel(pl.LightningModule):
    def __init__(
        self,
        test_df,
        model_name=CFG.model_name,
        output_dim=CFG.target_size,
        pretrained=True,
    ):
        super().__init__()

        self.test_df = test_df

        if model_name in timm.list_models(pretrained=True):
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
        else:
            raise NotImplementedError(f"{model_name} is not available in list {timm.list_models(pretrained=True)}")
        n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, output_dim)

    def binary_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

    def macro_auc(self, labels, pred):
        aucs = []
        for i in range(CFG.target_size):
            try:
                aucs.append(roc_auc_score(labels[:, i], pred[:, i]))
            except ValueError:
                aucs.append(0)
        return np.mean(aucs)

    def plot_macro_auc(self, labels, pred):
        fig, ax = plt.subplots(figsize=(8, 5))
        aucs = []
        for i, col in enumerate(CFG.target_cols):
            fpr, tpr, threshold = roc_curve(labels[:, i], pred[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, label=f"Field {col} (AUC = {roc_auc:.4f})")

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot([0, 1], [0, 1], label="Luck", linestyle="--", color="r")
        ax.plot(mean_auc, label=f"Average AUC score: {mean_auc:.4f} $\pm$ {std_auc:.4f}")
        ax.legend(loc="lower right")
        ax.set(
            xlim=[-0.1, 1.1],
            ylim=[-0.1, 1.1],
            title=f"Average AUC over {CFG.target_size} fields",
        )
        plt.show()
        return mean_auc

    def forward(self, x):
        bs = x.size(0)
        features = self.backbone(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output

    def training_step(self, train_batch, batch_idx):

        x, y = train_batch
        logits = self.forward(x)
        loss = self.binary_loss(logits, y).unsqueeze(0)
        y_hat = torch.sigmoid(logits)

        return {
            "loss": loss,
            "y": y.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
        }

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.binary_loss(logits, y).unsqueeze(0)
        y_hat = torch.sigmoid(logits)
        #  NOTE: already call torch.no_grad() so we had not to call the detach()
        return {
            "val_loss": loss,
            "y": y.cpu().numpy(),
            "y_hat": y_hat.cpu().numpy(),
        }

    def train_epoch_end(self, outputs):
        np.mean([out["loss"].item() for out in outputs])
        y_hat = np.concatenate([out["y_hat"] for out in outputs], axis=0)
        y = np.concatenate([out["y"] for out in outputs], axis=0)
        avg_auc = self.macro_auc(y, y_hat)
        print(f"EPOCH: {self.current_epoch} TRAIN AUC:{avg_auc:.4f}")

    def validation_epoch_end(self, outputs):
        np.mean([out["val_loss"].item() for out in outputs])
        y_hat = np.concatenate([out["y_hat"] for out in outputs], axis=0)
        y = np.concatenate([out["y"] for out in outputs], axis=0)
        avg_auc = self.macro_auc(y, y_hat)
        print(f"EPOCH: {self.current_epoch} VALIDATION AUC:{avg_auc:.4f}")

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        probs = torch.sigmoid(logits)
        return {"probs": probs.cpu().numpy()}

    def test_epoch_end(self, outputs):
        test_pred = np.concatenate([x["probs"] for x in outputs], axis=0)
        self.test_df[CFG.target_cols] = test_pred
        os.makedirs(CFG.submission_dir, exist_ok=True)
        N = len(os.listdir(CFG.submission_dir)) + 1
        self.test_df.to_csv(os.path.join(CFG.submission_dir, f"submission{N}.csv"), index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
