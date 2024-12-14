import metrics

import numpy as np
import torch

from torch import nn
from transformers import BertModel
from tqdm import tqdm

def weighted_mae(y_true, y_pred, num_preds):
    coords = y_pred[:, :, :2]  # Assuming coords are the first two columns
    weights = y_pred[:, :, 2:]  # Assuming weights are after the first two columns

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



class MultitaskBERTModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-cased', metadata_dim=6, num_preds=5, hidden_dim=256):
        super(MultitaskBERTModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion_layer = nn.Linear(self.bert.config.hidden_size + hidden_dim, hidden_dim)
        self.dense_layer = nn.Linear(hidden_dim, hidden_dim)

        self.coord_output = nn.Linear(hidden_dim, num_preds * 2)
        self.weight_output = nn.Linear(hidden_dim, num_preds)

    def forward(self, kf_input, kf_mask, metadata, num_preds):
        kf_embeddings = self.bert(input_ids=kf_input, attention_mask=kf_mask).pooler_output
        metadata_embeddings = self.metadata_encoder(metadata.float())

        fused_features = torch.cat([kf_embeddings, metadata_embeddings], dim=1)
        fused_output = self.fusion_layer(fused_features)
        dense_output = self.dense_layer(fused_output)

        coords = self.coord_output(dense_output).view(-1, num_preds, 2)
        weights = torch.softmax(self.weight_output(dense_output), dim=1).unsqueeze(-1)
        combined_output = torch.cat([coords, weights], dim=-1)

        return combined_output
    
def train_model(model, train, val, num_preds=5, epochs=3, batch_size=16, learning_rate=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_dataloader = torch.utils.data.DataLoader(list(train), batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(list(val), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader, desc="Training"):
            text_input, text_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            metadata, targets = batch['metadata'].to(device), batch['targets'].to(device)

            optimizer.zero_grad()
            predictions = model(text_input, text_mask, metadata, num_preds)
            loss = weighted_mae(targets, predictions, num_preds)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_loss_history.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss:.4f}\n")

        avg_val_loss = validate_model(model, val_dataloader, num_preds, device)
        val_loss_history.append(avg_val_loss)

    return train_loss_history, val_loss_history

def validate_model(model, val_dataloader, num_preds, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            text_input, text_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            metadata, targets = batch['metadata'].to(device), batch['targets'].to(device)

            predictions = model(text_input, text_mask, metadata, num_preds)
            loss = weighted_mae(targets, predictions, num_preds)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation  Loss: {avg_val_loss:.4f}")
    print("-" * 30)

    return avg_val_loss

def test_model(model, test, num_preds, batch_size):
    dataloader = torch.utils.data.DataLoader(list(test), batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            text_input, text_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            metadata, targets = batch['metadata'].to(device), batch['targets'].to(device)

            predictions = model(text_input, text_mask, metadata, num_preds)
            predictions = metrics.get_best_point(predictions.cpu().numpy())

            all_predictions.append(predictions)
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = metrics.get_results(all_predictions, all_targets)
    metrics.print_metrics(results)

    return results