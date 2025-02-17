import os
import pickle
from torch import Tensor
from torch.utils.data import DataLoader
from dataset_handler.data_loader_fetcher import custom_collate_fn
from dataset_handler.dataset_fetcher import DatasetType, split_dataset_fetcher
from dataset_handler.transformer_dataset import TransformerDataset, DatasetItem
from preprocessing.single_hop_multi_process import prepare_hit_k
from utils.data_saver import save


def from_split_to_hit_k_data(
        base_path: str,
        dataset_name: DatasetType,
        hop: int, max_num_neg: int = 500,
        alter_labeling: bool = False,
        num_workers: int = 10,
        batch_size: int = 32,
        inclusion_map: list[int] = None,
        subset: int = -1
) -> None:
    _, _, test = split_dataset_fetcher(path=base_path, dataset_name=dataset_name)

    bow_emb: Tensor = test.x
    test.edge_label_index = test.edge_label_index[:, test.edge_label == 1]

    if subset > 0:
        test.edge_label_index = test.edge_label_index[:, :subset]

    prepare_hit_k(
        test_data=test, hop=hop,
        max_num_neg=max_num_neg,
        dataset_name=dataset_name,
        alter_labeling=alter_labeling,
        num_workers=num_workers,
        base_path=base_path,
        inclusion_map=inclusion_map
    )

    base_hits_path = os.path.join(base_path, dataset_name.name, "HITS")
    processed_hits_path = os.path.join(base_hits_path, "dataloaders")

    print("YoUr AsS iS SaFe NoW, ThE HiTs ArE sAvEd, PrOcEeD To CrEaTe DaTaLoAdErS")

    for idx, filename in enumerate(os.listdir(base_hits_path)):
        dir_path = os.path.join(base_hits_path, filename)
        for split_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, split_name)
            with open(file_path, "rb") as f:
                res = pickle.load(f)

            neg_res: list[DatasetItem] = []
            sub_graph_res: list[Tensor] = []
            for idj, neg_samples in enumerate(res):
                negative_set, sub_graph = neg_samples
                for neg, sub in zip(negative_set, sub_graph):
                    neg_res.append(neg)
                    sub_graph_res.append(sub)

            processed_samples_dataset = TransformerDataset(
                alter_labeling=alter_labeling, mp=-1,
                pre_computed_set=(neg_res, sub_graph_res, bow_emb)
            )

            processed_samples_data_loader = DataLoader(
                dataset=processed_samples_dataset, batch_size=batch_size,
                collate_fn=custom_collate_fn, num_workers=num_workers,
                persistent_workers=True
            )

            full_path = os.path.join(processed_hits_path, f"{idx}", "dataloader_01.pt")
            save(processed_samples_data_loader, full_path)


# Use case
"""
dataset_name: DatasetType = DatasetType.CORA
hop: int = 2
alter_labeling: bool = True
from_split_to_hit_k_data(
        base_path=os.getcwd(),
        dataset_name=dataset_name,
        hop=hop,
        alter_labeling=alter_labeling,
) 
"""
