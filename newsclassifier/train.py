import gc
import os
import time
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from newsclassifier.config.config import Cfg, logger
from newsclassifier.data import (NewsDataset, collate, data_split,
                                 load_dataset, preprocess)
from newsclassifier.models import CustomModel
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(train_loader: DataLoader, model, num_classes: int, loss_fn, optimizer, epoch: int) -> float:
    """Train step."""
    model.train()
    loss = 0.0
    total_iterations = len(train_loader)
    desc = f"Training - Epoch {epoch+1}"
    for step, (inputs, labels) in tqdm(enumerate(train_loader), total=total_iterations, desc=desc):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  # reset gradients
        y_pred = model(inputs)  # forward pass
        targets = F.one_hot(labels.long(), num_classes=num_classes).float()  # one-hot (for loss_fn)
        J = loss_fn(y_pred, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (step + 1)  # cumulative loss
    return loss


def eval_step(val_loader: DataLoader, model, num_classes: int, loss_fn, epoch: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """Eval step."""
    model.eval()
    loss = 0.0
    total_iterations = len(val_loader)
    desc = f"Validation - Epoch {epoch+1}"
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for step, (inputs, labels) in tqdm(enumerate(val_loader), total=total_iterations, desc=desc):
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            y_pred = model(inputs)
            targets = F.one_hot(labels.long(), num_classes=num_classes).float()  # one-hot (for loss_fn)
            J = loss_fn(y_pred, targets).item()
            loss += (J - loss) / (step + 1)
            y_trues.extend(targets.cpu().numpy())
            y_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop(config=None):
    # ====================================================
    # loader
    # ====================================================

    config = dict(
        batch_size=Cfg.batch_size,
        num_classes=Cfg.num_classes,
        epochs=Cfg.epochs,
        dropout_pb=Cfg.dropout_pb,
        learning_rate=Cfg.lr,
        lr_reduce_factor=Cfg.lr_redfactor,
        lr_reduce_patience=Cfg.lr_redpatience,
    )

    with wandb.init(project="NewsClassifier", config=config):
        config = wandb.config

        df = load_dataset(Cfg.dataset_loc)
        ds, headlines_df, class_to_index, index_to_class = preprocess(df)
        train_ds, val_ds = data_split(ds, test_size=Cfg.test_size)

        logger.info("Preparing Data.")

        train_dataset = NewsDataset(train_ds)
        valid_dataset = NewsDataset(val_ds)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        # ====================================================
        # model
        # ====================================================

        logger.info("Creating Custom Model.")
        num_classes = config.num_classes
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

        min_loss = np.inf
        logger.info("Staring Training Loop.")
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

                if min_loss > val_loss:
                    min_loss = val_loss
                    print("Best Score : saving model.")
                    os.makedirs(Cfg.artifacts_path, exist_ok=True)
                    model.save(Cfg.artifacts_path)
                print(f"\nSaved Best Model Score: {min_loss:.4f}\n\n")
            except Exception as e:
                logger.error(f"Epoch - {epoch+1}, {e}")

        wandb.save(os.path.join(Cfg.artifacts_path, "model.pt"))
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train_loop()
