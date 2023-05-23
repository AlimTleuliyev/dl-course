# Unit 5.5. Organizing Your Data Loaders with Data Modules

import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import pandas as pd
import torch
from shared_utilities import LightningModel, MNISTDataModule, PyTorchMLP

train_val_dif = []
class MyCallback(Callback):
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module):
        train_acc = trainer.callback_metrics["train_acc"]
        val_acc = trainer.callback_metrics["val_acc"]
        dif = abs(train_acc - val_acc)
        print(f"\nDifference between train and val accuracy: {dif}")
        train_val_dif.append(dif)



if __name__ == "__main__":

    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    dm = MNISTDataModule()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cpu", devices=1, deterministic=True,
        logger=CSVLogger(save_dir="logs/", name="my-model"),
        callbacks=[MyCallback()]
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
    val_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

# Plotting the logs
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv") #type: ignore

aggreg_metrics = []
agg_col = "epoch"
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i #type: ignore
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[["train_loss", "val_loss"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
)

plt.savefig("loss-history.png")

df_metrics[["train_acc", "val_acc"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
)

plt.savefig("acc-history.png")

plt.figure()
plt.plot(train_val_dif)
plt.xlabel("Epoch")
plt.ylabel("Difference")
plt.savefig("train-val-dif.png")

