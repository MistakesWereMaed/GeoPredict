import pandas as pd
import numpy as np
import cudf
import torch
import faiss

from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def join_gpu(group):
    combined_row = {}
    
    for col in group.columns:
        if group[col].dtype == 'object':
            unique_values = group[col].dropna().unique().to_arrow().to_pylist()
            combined_row[col] = ', '.join(map(str, unique_values))
        elif cudf.api.types.is_numeric_dtype(group[col]):
            if col in {'latitude', 'longitude'}:
                combined_row[col] = group[col].mean()
            elif col in 'index':
                combined_row[col] = group[col].iloc[0]
            else:
                combined_row[col] = group[col].sum()
                
    return combined_row

def join_cpu(group):
    combined_row = {}
    
    for col in group.columns:
        if group[col].dtype == 'object':
            unique_values = group[col].dropna().unique()
            combined_row[col] = ', '.join(map(str, unique_values))
        elif pd.api.types.is_numeric_dtype(group[col]):
            if col in {'latitude', 'longitude'}:
                combined_row[col] = group[col].mean()
            elif col in 'index':
                combined_row[col] = group[col].iloc[0]
            else:
                combined_row[col] = group[col].sum()
    
    return combined_row

def join_rows(df, col, use_gpu=True):
    grouped = df.groupby(col)
    combined_rows = []

    if use_gpu:
        for _, group in tqdm(grouped, desc="Processing Groups", total=len(grouped)):
            combined_rows.append(join_gpu(group))
        return cudf.DataFrame(combined_rows)
    
    for _, group in grouped:
        combined_rows.append(join_cpu(group))
    return pd.DataFrame(combined_rows)

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

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
    def __init__(self, input_ids, attention_mask, metadata, targets):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.metadata = metadata
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'metadata': self.metadata[idx],
            'targets': self.targets[idx]
        }

def add_random_nans(metadata, nan_percentage):
    metadata_np = metadata.numpy()
    total_elements = metadata_np.size
    num_nans = int(total_elements * nan_percentage / 100)

    flat_indices = np.random.choice(total_elements, size=num_nans, replace=False)
    np.put(metadata_np, flat_indices, np.nan)

    metadata_with_nans = torch.tensor(metadata_np, dtype=metadata.dtype)
    return metadata_with_nans

def impute_metadata(dataset, n_neighbors=5, nan_percentage=1):
    # Extract tensors from the dataset
    input_ids = dataset.input_ids
    attention_mask = dataset.attention_mask
    metadata = torch.tensor(dataset.metadata, dtype=torch.float32)
    targets = torch.tensor(dataset.targets, dtype=torch.float64)

    # Add random nans to the metadata
    metadata = add_random_nans(metadata, nan_percentage)

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
        return dataset

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
    return GeolocationDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        metadata=metadata,
        targets=targets
    )

def load_data(df, labeled=True):
    df_key, df_metadata, df_y = split_features(df, labeled=labeled)

    tokens = tokenize(df_key['text'].astype(str).tolist())
    df_metadata = encode_df(df_metadata)
    df_metadata = scale_df(df_metadata)

    dataset = GeolocationDataset(
        input_ids = tokens['input_ids'],
        attention_mask = tokens['attention_mask'],
        metadata = df_metadata,
        targets = df_y.to_numpy()
    )

    return dataset