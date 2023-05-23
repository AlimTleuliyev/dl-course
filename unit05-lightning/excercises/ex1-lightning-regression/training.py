# Unit 5.5. Organizing Your Data Loaders with Data Modules

import lightning as L
import torch
from shared_utilities import LightningModel, PyTorchMLPRegressor, AmesHousingDataModule
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    dm = AmesHousingDataModule()
    

    pytorch_model = PyTorchMLPRegressor(num_features=3)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.001)

    trainer = L.Trainer(
        max_epochs=100, 
        accelerator="cpu",
        devices="auto",
        logger=CSVLogger(save_dir="logs/", name="my-model"),
        deterministic=True,
    )
    trainer.fit(model=lightning_model, datamodule=dm)
    

    train_mse = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_mse"]
    val_mse = trainer.validate(datamodule=dm)[0]["val_mse"]
    test_mse = trainer.test(datamodule=dm)[0]["test_mse"]
    print(
        f"Train mse {train_mse:.2f}"
        f" | Val mse {val_mse:.2f}"
        f" | Test mse {test_mse:.2f}"
    )


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
df_metrics[["train_mse", "val_mse"]].plot(
    grid=True, legend=True, xlabel="Epoch", ylabel="MSE"
)

plt.savefig("mse-history.png")