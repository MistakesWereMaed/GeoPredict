from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import MultitaskBERTModel
from dataloader import load_data

PATH_TRAIN = '../Data/Training/train.csv'
PATH_DEV = '../Data/Training/dev.csv'

NUM_PREDS = 3
HIDDEN_DIM = 128
LR = 0.006
BATCH_SIZE = 128



def main():
    logger = WandbLogger(project="GeoPredict")

    checkpoint_cb = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    model = MultitaskBERTModel(
        num_preds=NUM_PREDS,
        hidden_dim=HIDDEN_DIM,
        lr=LR
    )

    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        strategy="ddp",
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    train_loader = load_data(PATH_TRAIN, batch_size=BATCH_SIZE)
    val_loader = load_data(PATH_DEV, batch_size=BATCH_SIZE, shuffle=False)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()