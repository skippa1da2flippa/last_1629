import os
from typing import Tuple, Union
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from dataset_handler.dataset_fetcher import dataset_split_max_size, DatasetType
from dataset_handler.transformer_dataset import TransformerDataset, DatasetItem


def custom_collate_fn(batch: list[DatasetItem]) -> DatasetItem:
    bow_emb: list[Tensor] = [item.bow_emb for item in batch]
    pos_emb: list[Tensor] = [item.pos_emb for item in batch]
    mp_emb: list[Union[Tensor, None]] = [item.mp_emb for item in batch]
    target_footprints: list[Tensor] = [item.target_footprint for item in batch]
    altered_labels: list[Union[Tensor, None]] = [item.altered_labels for item in batch]

    # Determine the maximum length for padding
    max_len: int = max(x.size(0) for x in bow_emb)

    # Pad each tensor along the sequence dimension
    bow_emb_padded: Tensor = pad_sequence(bow_emb, batch_first=True)  # Shape: (batch_size, max_len, bow_emb_dim)
    pos_emb_padded: Tensor = pad_sequence(pos_emb, batch_first=True)  # Shape: (batch_size, max_len, pos_emb_dim)
    target_footprints_padded: Tensor = pad_sequence(target_footprints, batch_first=True)  # Shape: (batch_size, max_len)

    # Create the padding mask for the attention mechanism
    mask: Tensor = torch.zeros(len(bow_emb), max_len, dtype=torch.bool)
    for i, length in enumerate([x.size(0) for x in bow_emb]):
        mask[i, length:] = True  # Mask padded positions

    labels: Tensor = torch.stack([item.labels for item in batch])

    if mp_emb[0] is not None:
        mp_emb_padded: Tensor = pad_sequence(mp_emb, batch_first=True)  # Shape: (batch_size, max_len, mp_emb_dim)
    else:
        mp_emb_padded: Tensor = torch.tensor([])

    if altered_labels[0] is not None:
        altered_labels_padded: Tensor = pad_sequence(altered_labels, batch_first=True)
    else:
        altered_labels_padded: Tensor = torch.tensor([])

    return DatasetItem(
        bow_emb=bow_emb_padded,
        pos_emb=pos_emb_padded,
        mp_emb=mp_emb_padded,
        mask=mask,
        labels=labels,
        target_footprint=target_footprints_padded,
        altered_labels=altered_labels_padded
    )


def transformer_prepare_data_loaders(
        path: str, hop: int, mp: int,
        batch_size: int = 32,
        limit: int = -1,
        prepare_test: bool = False,
        device: str = "cpu",
        max_nodes: int = -1,
        alter_labeling: bool = False,
        num_workers: int = 4,
        test_shuffle: bool = True,
        just_test_or_val: bool = False,
        train_split: list[int] = None,
        val_split: list[int] = None,
        test_split: list[int] = None,
        no_pos: bool = False,
        local_pos: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:

    dataset_name: str = os.path.basename(path)
    dataset_value: str = DatasetType[dataset_name].value
    train: TransformerDataset = TransformerDataset(
        dataset_path=path, mp=mp,
        data_type="train_dataset",
        hop=hop, device=device,
        limit=limit, max_nodes=max_nodes,
        alter_labeling=alter_labeling,
        fill_empty=just_test_or_val,
        get_split=train_split,
        no_pos=no_pos,
        local=local_pos
    )

    val: TransformerDataset = TransformerDataset(
        dataset_path=path, mp=mp,
        max_nodes=dataset_split_max_size[dataset_value][hop - 1],
        hop=hop, data_type="val_dataset",
        device=device,
        alter_labeling=alter_labeling,
        get_split=val_split,
        no_pos=no_pos,
        local=local_pos
    )

    if prepare_test:
        test = TransformerDataset(
            dataset_path=path, mp=mp,
            max_nodes=dataset_split_max_size[dataset_value][hop - 1],
            hop=hop, data_type="test_dataset",
            device=device, alter_labeling=alter_labeling,
            get_split=test_split, no_pos=no_pos,
            local=local_pos
        )

        test_data_loader = DataLoader(
            dataset=test, batch_size=batch_size, shuffle=test_shuffle,
            collate_fn=custom_collate_fn, num_workers=num_workers
        )

        train.extend(val)
        train_val_data_loader = DataLoader(
            dataset=train, batch_size=batch_size, shuffle=True,
            collate_fn=custom_collate_fn, num_workers=num_workers
        )

        return train_val_data_loader, test_data_loader

    else:
        val_data_loader = DataLoader(
            dataset=val, batch_size=batch_size, shuffle=True,
            collate_fn=custom_collate_fn, num_workers=num_workers,
            persistent_workers=True
        )

        if not just_test_or_val:
            train_data_loader = DataLoader(
                dataset=train, batch_size=batch_size, shuffle=True,
                collate_fn=custom_collate_fn, num_workers=num_workers,
                persistent_workers=True
            )

            return train_data_loader, val_data_loader

        else:
            return val_data_loader
