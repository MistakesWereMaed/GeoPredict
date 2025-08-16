import pandas as pd
import numpy as np
import torch
import faiss

from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader



def encode_df(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_df(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df

def split_features(df, labeled=True):
    df_key = pd.DataFrame(df['text'])
    df_meta = df[['hour', 'weekday', 'name', 'population', 'latitude_y', 'longitude_y']]
    df_y = df[['latitude_x', 'longitude_x']] if labeled else None

    return df_key, df_meta, df_y

def tokenize(text, length=200):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    tokens = tokenizer(
        text,
        add_special_tokens=True,
        max_length=length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask']
    }

class GeolocationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, metadata, targets, device):
        self.input_ids = input_ids.to(device, non_blocking=True)
        self.attention_mask = attention_mask.to(device, non_blocking=True)
        self.metadata = metadata.to(device, non_blocking=True)
        self.targets = torch.as_tensor(targets, device=device)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'metadata': self.metadata[idx],
            'targets': self.targets[idx]
        }
    
def impute_metadata(input_ids, attention_mask, metadata, targets, n_neighbors=5):
    # Combine all features into a single tensor for KNN processing
    full_features = torch.cat([input_ids, attention_mask, metadata, targets], dim=1)
    # Convert to NumPy for FAISS processing
    full_features_np = full_features.numpy()
    metadata_np = metadata.numpy()
    # Identify rows with and without missing metadata
    missing_mask = np.isnan(metadata_np).any(axis=1)
    non_missing_mask = ~missing_mask
    # If there are no missing metadata rows, return the original dataset
    if not missing_mask.any():
        print("No missing metadata found. Skipping imputation.")
        return metadata
    # Split the dataset into rows with complete and missing metadata
    complete_data = full_features_np[non_missing_mask]
    incomplete_data = full_features_np[missing_mask]
    # Set up FAISS index using complete rows
    index = faiss.IndexFlatL2(complete_data.shape[1])  # L2 (Euclidean) distance
    index.add(complete_data)
    # Perform KNN search for rows with missing metadata
    _, neighbor_indices = index.search(incomplete_data, n_neighbors)
    # Impute missing metadata by averaging nearest neighbors' metadata
    imputed_metadata = []
    metadata_complete = metadata_np[non_missing_mask]
    for row_idx, neighbors in enumerate(neighbor_indices):
        # Compute mean of the neighbors' metadata
        neighbor_metadata = metadata_complete[neighbors]
        imputed_row = np.nanmean(neighbor_metadata, axis=0)
        imputed_metadata.append(imputed_row)
    # Replace missing metadata in the original array
    imputed_metadata = np.array(imputed_metadata)
    metadata_np[missing_mask] = imputed_metadata
    # Reconstruct the metadata tensor
    metadata = torch.tensor(metadata_np, dtype=torch.float32)
    # Return a new dataset with updated metadata
    return metadata

def load_data(path, batch_size, labeled=True, shuffle=True):
    df = pd.read_csv(path)
    df_key, df_metadata, df_y = split_features(df, labeled=labeled)

    tokens = tokenize(df_key['text'].astype(str).tolist())
    df_metadata = encode_df(df_metadata)
    df_metadata = scale_df(df_metadata)

    targets = torch.tensor(df_y.to_numpy(), dtype=torch.float32)
    metadata = torch.tensor(df_metadata, dtype=torch.float32)
    metadata = impute_metadata(tokens['input_ids'], tokens['attention_mask'], metadata, targets)

    device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    dataset = GeolocationDataset(
        input_ids = tokens['input_ids'],
        attention_mask = tokens['attention_mask'],
        metadata = metadata,
        targets = targets,
        device = device
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader