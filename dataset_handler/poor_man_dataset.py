import gc
import os
import random
import lightning as pl
from torch.utils.data import DataLoader
from dataset_handler.data_loader_fetcher import custom_collate_fn
from dataset_handler.dataset_fetcher import dataset_split_max_size, DatasetType
from dataset_handler.transformer_dataset import TransformerDataset

"""
    No bill gates here I have 32 GB of ram PUBMED train is almost 80 GB
    I splitted the dataset in 10 so taking into consideration my hardware 
    I can load at most two splits at a time in memory. This means having 
    to improve the number of epochs to: actual_epochs * (n_split // split_per_time).
    Furthermore the trainer should have the flag  
    check_val_every_n_epoch = n_split // split_per_time
"""


class CustomDataModule(pl.LightningDataModule):
    def __init__(
            self, dataset_path: str,
            hop: int,
            max_nodes: int = -1,
            batch_size: int = 32,
            n_split: int = 10,
            num_splits_per_epoch: int = 2,
            alter_labeling: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path: str = dataset_path
        self.dataset_name: str = DatasetType[os.path.basename(self.dataset_path)].value
        self.batch_size: int = batch_size
        self.hop: int = hop
        self.alter_labeling: bool = alter_labeling
        self.max_nodes: int = max_nodes
        self.pos_split_path = [x for x in range(n_split // 2)]
        self.neg_split_path = [x for x in range(n_split // 2, n_split)]
        self.num_splits_per_epoch: int = num_splits_per_epoch
        self.epoch_counter: int = 0
        self.total_epochs_per_cycle = n_split // self.num_splits_per_epoch  # 5 epochs in this case
        self.train_dataloader_instance: DataLoader = None
        self.val_dataloader_instance: DataLoader = None
        self.pos_idx: int = 0
        self.neg_idx: int = 0

        self._shuffle_splits()

    def _shuffle_splits(self) -> None:
        """Shuffle dataset splits and reset tracking."""
        print("I'M SHUFFLING")
        random.shuffle(self.pos_split_path)  # Shuffle at the start of a cycle
        random.shuffle(self.neg_split_path)
        self.epoch_counter = 0
        self.pos_idx = self.neg_idx = 0

    def setup(self, stage: str = None) -> None:
        print("START OF THE TRAINING")
        """Called once at the beginning of training."""
        if stage == "fit":
            self._init_val_dataloader()

    def _init_val_dataloader(self) -> None:
        """Load full validation dataset (static, no dynamic splits)."""
        val_dataset = TransformerDataset(
            dataset_path=self.dataset_path, hop=self.hop,
            max_nodes=dataset_split_max_size[self.dataset_name][self.hop - 1],
            alter_labeling=self.alter_labeling, data_type="val_dataset",
            get_split=[x for x in range(5)]
        )

        self.val_dataloader_instance = DataLoader(
            dataset=val_dataset, batch_size=self.batch_size,
            num_workers=4, collate_fn=custom_collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        print("Just called val_dataloader method")
        """Validation DataLoader (static dataset)."""
        return self.val_dataloader_instance

    def train_dataloader(self) -> DataLoader:
        print("Just called train_dataloader method")
        """Called at the start of every epoch by the Trainer."""
        self._update_dataset()  # Initialize the first dataset
        print("Just called update_dataset")
        return self.train_dataloader_instance

    def _update_dataset(self) -> None:
        """Update dataset splits at each epoch."""
        if self.epoch_counter >= self.total_epochs_per_cycle:
            print("JUST FINISHED A REAL EPOCH")
            self._shuffle_splits()  # Reshuffle after all splits are used

        split: list[int] = [
            self.pos_split_path[self.pos_idx],
            self.neg_split_path[self.neg_idx]
        ]

        self.pos_idx += 1
        self.neg_idx += 1

        self.train_dataloader_instance = None
        gc.collect()

        dataset = TransformerDataset(
            dataset_path=self.dataset_path, hop=self.hop,
            max_nodes=dataset_split_max_size[self.dataset_name][self.hop - 1],
            alter_labeling=self.alter_labeling, get_split=split
        )

        self.train_dataloader_instance = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, collate_fn=custom_collate_fn, persistent_workers=True
        )

        self.epoch_counter += 1  # Move to the next set of splits


# trainer.fit(model, datamodule=data_module) call the trainer like this

