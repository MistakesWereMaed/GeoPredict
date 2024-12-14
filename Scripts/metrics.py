import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial import cKDTree

PATH_STATE_CITY = '../Data/Unprocessed/state_city.txt'

def haversine_distance(y_true, y_pred):
    # Radius of the Earth in kilometers
    R = 6371.0
    pi = np.pi

    # Convert degrees to radians
    lat_true = y_true[:, 0] * (pi / 180.0)
    lon_true = y_true[:, 1] * (pi / 180.0)
    lat_pred = y_pred[:, 0] * (pi / 180.0)
    lon_pred = y_pred[:, 1] * (pi / 180.0)

    # Compute the differences
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true

    # Apply the Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat_true) * np.cos(lat_pred) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c  # Distance in kilometers

    return distance

def find_nearest_city_state(tree, df_state_city, coordinates):
    dist, idx = tree.query(coordinates, k=1)
    nearest_city_state = df_state_city.iloc[idx]
    return str(nearest_city_state['city']) + ', ' + str(nearest_city_state['state'])

def calculate_classification_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return precision, recall, f1

def get_metrics(results):
    precision, recall, f1 = calculate_classification_metrics(results['true_city_state'].astype(str), results['pred_city_state'].astype(str))
    metrics = {
        "Mean SAE": results['distances'].mean(),
        "Median SAE": results['distances'].median(),
        "Acc@161": ((results['distances'] < 161).sum() / len(results)) * 100,
        "Precision": precision, 
        "Recall": recall, 
        "F1 Score": f1
    }

    return metrics

def print_metrics(results):
    metrics = get_metrics(results)
    print("-------------------------------------")
    print("Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    return metrics

def get_results(predictions, y_true):
    state_city_df = pd.read_csv(PATH_STATE_CITY, delimiter='\t')
    tree = cKDTree(state_city_df[['latitude', 'longitude']].values)
    results = pd.DataFrame()

    y_pred = np.array(predictions).reshape(-1, 2)

    results['distances'] = haversine_distance(y_true, y_pred)
    results['lat_true'] = list(y_true[:, 0])
    results['lon_true'] = list(y_true[:, 1])
    results['lat_pred'] = list(y_pred[:, 0])
    results['lon_pred'] = list(y_pred[:, 1])
    results['true_city_state'] = [find_nearest_city_state(tree, state_city_df, true) for true in list(y_true)]
    results['pred_city_state'] = [find_nearest_city_state(tree, state_city_df, pred) for pred in list(y_pred)]
    
    return results

def get_best_point(predictions):
    coords = predictions[:, :, :2]
    weights = predictions[:, :, 2]
    # Find the index of the maximum weight for each sample in the batch
    max_weight_indices = np.argmax(weights, axis=1)  # Shape: (batch,)
    # Use the indices to gather the corresponding coordinate pairs
    highest_weighted_points = np.take_along_axis(
        coords, max_weight_indices[:, np.newaxis, np.newaxis], axis=1
    ).squeeze(axis=1)  # Shape: (batch, 2)
    
    return highest_weighted_points