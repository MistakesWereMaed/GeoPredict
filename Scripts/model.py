import torch
import pytorch_lightning as pl

from torch import nn
from transformers import AutoModel

METADATA_DIM = 6



def weighted_mae(y_true, y_pred, num_preds):
    coords = y_pred[:, :, :2]
    weights = y_pred[:, :, 2:]
    # US boundaries for penalty
    lat_min, lat_max = 24.396308, 49.384358
    lon_min, lon_max = -125.0, -66.93457
    # Compute the mean absolute error (MAE)
    y_true_expanded = y_true.unsqueeze(1).expand(-1, num_preds, -1)
    mae = torch.abs(y_true_expanded - coords).mean(dim=-1)
    # Initialize penalty as zeros
    penalty = torch.zeros(coords.shape[0], dtype=torch.float32).to(coords.device)
    # Apply penalty for coordinates outside of US bounds
    penalty += torch.sum((coords[:, :, 0] < lat_min) | (coords[:, :, 0] > lat_max), dim=-1)
    penalty += torch.sum((coords[:, :, 1] < lon_min) | (coords[:, :, 1] > lon_max), dim=-1)
    # Compute the weighted loss
    weighted_loss = (mae * weights.squeeze(-1)).sum(dim=1)
    # Add penalty to the weighted loss
    loss = weighted_loss + penalty

    return loss.mean()

class MultitaskBERTModel(pl.LightningModule):
    def __init__(self, num_preds=5, hidden_dim=256, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        self.metadata_encoder = nn.Sequential(
            nn.Linear(METADATA_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fusion_layer = nn.Linear(self.bert.config.hidden_size + hidden_dim, hidden_dim)
        self.dense_layer = nn.Linear(hidden_dim, hidden_dim)

        self.coord_output = nn.Linear(hidden_dim, num_preds * 2)
        self.weight_output = nn.Linear(hidden_dim, num_preds)

    def forward(self, kf_input, kf_mask, metadata):
        num_preds = self.hparams.num_preds
        outputs = self.bert(input_ids=kf_input, attention_mask=kf_mask)
        kf_embeddings = outputs.last_hidden_state[:, 0]
        metadata_embeddings = self.metadata_encoder(metadata)

        fused_features = torch.cat([kf_embeddings, metadata_embeddings], dim=1)
        fused_output = self.fusion_layer(fused_features)
        dense_output = self.dense_layer(fused_output)

        coords = self.coord_output(dense_output).view(-1, num_preds, 2)
        weights = torch.softmax(self.weight_output(dense_output), dim=1).unsqueeze(-1)
        combined_output = torch.cat([coords, weights], dim=-1)

        return combined_output
    
    def training_step(self, batch, batch_idx):
        predictions = self(batch['input_ids'], batch['attention_mask'], batch['metadata'])
        loss = weighted_mae(batch['targets'], predictions, self.hparams.num_preds)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch['input_ids'], batch['attention_mask'], batch['metadata'])
        loss = weighted_mae(batch['targets'], predictions, self.hparams.num_preds)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)