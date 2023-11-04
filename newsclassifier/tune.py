import gc
import time

import torch
import torch.nn as nn
import wandb
from newsclassifier.config.config import Cfg, logger
from newsclassifier.data import (NewsDataset, data_split, load_dataset,
                                 preprocess)
from newsclassifier.models import CustomModel
from newsclassifier.train import eval_step, train_step
from newsclassifier.utils import read_yaml
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tune_loop(config=None):
    """Tune loop."""
    # ====================================================
    # loader
    # ====================================================
    logger.info("Starting Tuning.")
    with wandb.init(project="NewsClassifier", config=config):
        config = wandb.config

        df = load_dataset(Cfg.dataset_loc)
        ds, headlines_df, class_to_index, index_to_class = preprocess(df)
        train_ds, val_ds = data_split(ds, test_size=Cfg.test_size)

        train_dataset = NewsDataset(train_ds)
        valid_dataset = NewsDataset(val_ds)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        # ====================================================
        # model
        # ====================================================
        num_classes = Cfg.num_classes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = CustomModel(num_classes=num_classes, dropout_pb=config.dropout_pb)
        model.to(device)

        # ====================================================
        # Training components
        # ====================================================
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_reduce_factor, patience=config.lr_reduce_patience
        )

        # ====================================================
        # loop
        # ====================================================
        wandb.watch(model, criterion, log="all", log_freq=10)

        for epoch in range(config.epochs):
            try:
                start_time = time.time()

                # Step
                train_loss = train_step(train_loader, model, num_classes, criterion, optimizer, epoch)
                val_loss, _, _ = eval_step(valid_loader, model, num_classes, criterion, epoch)
                scheduler.step(val_loss)

                # scoring
                elapsed = time.time() - start_time
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
                print(f"Epoch {epoch+1} - avg_train_loss: {train_loss:.4f}  avg_val_loss: {val_loss:.4f}  time: {elapsed:.0f}s")
            except Exception as e:
                logger.error(f"Epoch {epoch+1}, {e}")

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    sweep_config = read_yaml(Cfg.sweep_config_path)
    sweep_id = wandb.sweep(sweep_config, project="NewsClassifier")
    wandb.agent(sweep_id, tune_loop, count=Cfg.sweep_runs)
