import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from dataloader import load_data
from model import MultitaskBERTModel
from metrics import get_results, print_metrics



PATH_TEST = '../Data/Training/test.csv'
PATH_CHKPT = "./GeoPredict/zf6qesof/checkpoints/epoch=0-step=2520.ckpt"

NUM_PREDS = 3
HIDDEN_DIM = 128
LR = 0.006
BATCH_SIZE = 128



def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[1].scatter(results['lon_true'], results['lat_true'], c='blue', marker='o', alpha=0.3, label='True')
    axes[1].scatter(results['lon_pred'], results['lat_pred'], c='orange', marker='o', alpha=0.8, label='Predicted')
    axes[1].set_title('Results Latitude and Longitude Plot')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].legend()

    plt.tight_layout()
    plt.show(block=True)

def main():
    model = MultitaskBERTModel.load_from_checkpoint(checkpoint_path=PATH_CHKPT)
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1,
        logger=False
    )

    test_dataloader = load_data(PATH_TEST, batch_size=BATCH_SIZE, shuffle=False)
    outputs = trainer.predict(model, dataloaders=test_dataloader)

    all_predictions = np.concatenate([o["predictions"] for o in outputs], axis=0)
    all_targets = np.concatenate([o["targets"] for o in outputs], axis=0)
    results = get_results(all_predictions, all_targets)

    plot_results(results)
    print_metrics(results)

if __name__ == "__main__":
    main()