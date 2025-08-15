import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import MultitaskBERTModel
from dataloader import load_data

PATH_TRAIN = '../Data/Training/test.csv'
PATH_DEV = '../Data/Training/dev.csv'



def train(train_loader, val_loader):
    print("Initializing logger...")
    logger = WandbLogger(project="GeoPredict")

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    print("Initializing model...")
    model = MultitaskBERTModel(
        metadata_dim=6,
        num_preds=3,
        hidden_dim=330,
        lr=1e-4
    )

    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    trainer.fit(model, train_loader, val_loader)

def main():
    train_loader = load_data(PATH_TRAIN, batch_size=16)
    val_loader = load_data(PATH_DEV, batch_size=16, shuffle=False)

    train(train_loader, val_loader)

if __name__ == "__main__":
    main()