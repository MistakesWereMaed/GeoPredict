import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from model import MultitaskBERTModel
from dataloader import load_data

PATH_TRAIN = '../Data/Training/test.csv'
PATH_DEV = '../Data/Training/dev.csv'

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        logger = WandbLogger(project="GeoPredict", config=config)

        model = MultitaskBERTModel(
            num_preds=config.num_preds,
            hidden_dim=config.hidden_dim,
            lr=config.lr
        )

        train_loader = load_data(PATH_TRAIN, batch_size=128)
        val_loader = load_data(PATH_DEV, batch_size=128, shuffle=False)

        trainer = Trainer(
            max_epochs=2,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            logger=logger,
            limit_val_batches=300
        )

        trainer.fit(model, train_loader, val_loader)

def main():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "lr": {"min": 1e-4, "max": 1e-2},
            "hidden_dim": {"values": [128, 256, 512]},
            "num_preds": {"values": [1, 3]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="GeoPredict")
    wandb.agent(sweep_id, function=train, count=20)


if __name__ == "__main__":
    main()