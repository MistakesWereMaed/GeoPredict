import matplotlib.pyplot as plt
import seaborn as sns

def get_stats(df):
    cluster_stats = df.groupby('cluster').agg({
        'latitude': ['mean', 'median', 'std'],
        'longitude': ['mean', 'median', 'std'],
        'cluster': ['count']
    })

    cluster_stats.columns = ['_'.join(col) for col in cluster_stats.columns]
    cluster_stats = cluster_stats.reset_index()

    return cluster_stats

def show_clusters(ax, df):
    scatter = ax.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='tab20', s=10)
    num_clusters = len(df['cluster'].unique())
    handles, labels = [], []

    for cluster in range(num_clusters):
        handles.append(
            plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=scatter.cmap(scatter.norm(cluster)),
                markersize=10
            )
        )
        labels.append(f"Cluster {cluster}")

    ax.legend(handles=handles, labels=labels, title='Clusters', loc='upper left', bbox_to_anchor=(1, 1), title_fontsize='13')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographical Data Clustering (K-means)')

def show_count(ax, data):
    sns.barplot(x='cluster', y='cluster_count', data=data, color='purple', ax=ax)
    ax.set_ylabel("Row Count")
    ax.set_title("Row Count per Cluster")
    ax.set_xlabel("Cluster")

def show_stats(df, data):
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]}
    )

    show_clusters(axes[0], df)
    show_count(axes[1], data)

    plt.tight_layout()
    plt.show()

def show_training_history(train_loss_history, val_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss', linestyle='--')

    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_results(pfas, results):
    # Create a figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # First subplot: PFAS data
    show_clusters(axes[0], pfas)

    # Second subplot: Results data
    axes[1].scatter(results['lon_true'], results['lat_true'], c='blue', marker='o', alpha=0.3, label='True')
    axes[1].scatter(results['lon_pred'], results['lat_pred'], c='orange', marker='o', alpha=0.8, label='Predicted')
    axes[1].set_title('Results Latitude and Longitude Plot')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].legend()

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

def plot_regions(results, mean_distance):
    # Create a figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    show_clusters(results)

    plt.figure(figsize=(8, 5))
    mean_distance.plot(kind='bar', color='skyblue', alpha=0.8)

    # Add labels and title
    plt.title('Mean Distance by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Distance')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Show the plot
    plt.show()