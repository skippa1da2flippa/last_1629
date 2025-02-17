import os
import pickle
from typing import Any, NamedTuple, Tuple, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset


class DatasetItem(NamedTuple):
    bow_emb: Tensor
    pos_emb: Tensor
    mp_emb: Optional[Tensor]
    mask: Optional[Tensor]
    labels: Tensor
    target_footprint: Tensor
    altered_labels: Optional[Tensor]


class TransformerDataset(Dataset):
    def __init__(
            self, dataset_path: str = None,
            mp: int = -1,
            hop: int = None,
            max_nodes: int = -1,
            limit: int = -1,
            data_type: str = "train_dataset",
            alter_labeling: bool = False,
            pre_computed_set: Tuple[list[DatasetItem], list[Tensor], Tensor] = None,
            get_split: list[int] = None,
            fill_empty: bool = False,
            no_pos: bool = False,
            local: bool = True,
            device: str = "cpu"
    ) -> None:
        super().__init__()

        self.path: str = dataset_path
        self.max_nodes: int = max_nodes
        self.hop: int = hop
        self.data_type: str = data_type
        self.device: str = device
        self.mp: int = mp
        self.limit: int = limit
        self.alter_labeling: bool = alter_labeling
        self.no_pos: bool = no_pos
        self.local: bool = local

        self.BOW_emb: Tensor = torch.tensor([])
        self.pos_emb: list[Tensor] = []
        self.mp_emb: list[Tensor] = []
        self.labels: list[Tensor] = []
        self.target_footprint: list[Tensor] = []
        self.altered_labels: list[Tensor] = []
        self.sub_graphs_ids: list[Tensor] = []

        if not fill_empty:
            if pre_computed_set is None:
                if get_split is not None:
                    self.process_split(split=get_split)
                else:
                    self._preprocess()
            else:
                self._assign_data(data=pre_computed_set)

            if self.limit > 0:
                self.limit_one()

    def _preprocess(self) -> None:
        processed_path = os.path.join(self.path, f"processed_dataset\\{self.data_type}")
        target_labels_path = os.path.join(processed_path, "targets_labels.pt")
        with open(target_labels_path, "rb") as f:
            target_labels: dict[str, Any] = pickle.load(f)

        self.labels = target_labels["labels"]
        self.BOW_emb = target_labels["BOW"].to(self.device)

        hop_path = os.path.join(processed_path, f"hop_{self.hop}")
        pos_enc_path = os.path.join(hop_path, f"pos_enc.pt")
        subgraph_ids_path = os.path.join(hop_path, "subgraph_ids.pt")

        with open(pos_enc_path, "rb") as f:
            self.pos_emb = pickle.load(f)

        if self.mp >= 0:
            mp_enc_path = os.path.join(hop_path, f"mp_{self.mp}.pt")
            with open(mp_enc_path, "rb") as f:
                self.mp_emb = pickle.load(f)

        with open(subgraph_ids_path, "rb") as f:
            self.sub_graphs_ids = pickle.load(f)

        self._process_each_sample(targets_nodes=target_labels["target_nodes"])

    def _process_each_sample(self, targets_nodes: list[Tensor]) -> None:

        max_sub_graph_ids: Tensor = max(self.sub_graphs_ids, key=lambda x: x.numel())
        max_sub_graph_size: int = max_sub_graph_ids.numel()
        self.max_nodes = max_sub_graph_size if max_sub_graph_size > self.max_nodes else self.max_nodes

        for idx, (sub_graph, target_node) in enumerate(
                zip(self.sub_graphs_ids, targets_nodes)
        ):

            self.labels[idx] = self.labels[idx].to(self.device)

            u_mask = (sub_graph == target_node[0]).int()
            v_mask = (sub_graph == target_node[1]).int()
            uv_mask: Tensor = u_mask | v_mask

            self.target_footprint.append(uv_mask)

            if self.alter_labeling:
                altered_label: Tensor = (
                        self.pos_emb[idx] * uv_mask.unsqueeze(dim=0)
                ).mean(dim=1)
                altered_label[torch.nonzero(uv_mask, as_tuple=False).reshape(-1)] = 1

                self.altered_labels.append(altered_label)

            pad: Tensor = torch.full(
                size=(
                    sub_graph.shape[0],
                    self.max_nodes - sub_graph.shape[0]
                ),
                fill_value=0., device=self.device
            )

            if self.no_pos:
                if self.local:
                    self.pos_emb[idx] = torch.tensor(
                        [idx for idx in range(self.pos_emb[idx].shape[0])],
                        device=self.device
                    )
                else:
                    self.pos_emb[idx] = self.sub_graphs_ids[idx]
            else:
                self.pos_emb[idx] = self.pos_emb[idx].to(self.device)
                self.pos_emb[idx] = torch.cat([self.pos_emb[idx], pad], dim=-1)

            if self.mp >= 0:
                self.mp_emb[idx] = self.mp_emb[idx].to(self.device)

                if self.mp_emb[idx].shape[0] != self.mp_emb[idx].shape[1]:  # TODO REMOVE THIS
                    self.mp_emb[idx] = self.mp_emb[idx][sub_graph, :]

                self.mp_emb[idx] = torch.cat([self.mp_emb[idx], pad], dim=-1)

    def limit_one(self):
        # Ensure the limit does not exceed the list length
        limit = min(self.limit, len(self.pos_emb))

        # Create the limited versions by preserving values from both ends
        limited_pos_emb = self.pos_emb[:limit // 2] + self.pos_emb[-(limit // 2):]
        limited_labels = self.labels[:limit // 2] + self.labels[-(limit // 2):]
        limited_target_footprint = (
                self.target_footprint[:limit // 2] + self.target_footprint[-(limit // 2):]
        )
        limited_sub_graph_ids = self.sub_graphs_ids[:limit // 2] + self.sub_graphs_ids[-(limit // 2):]

        # Assign the limited lists back to the instance variables
        self.pos_emb = limited_pos_emb

        if self.mp >= 0:
            limited_mp_emb = self.mp_emb[:limit // 2] + self.mp_emb[-(limit // 2):]
            self.mp_emb = limited_mp_emb

        self.labels = limited_labels
        self.target_footprint = limited_target_footprint
        self.sub_graphs_ids = limited_sub_graph_ids

    def extend(self, sdn_dataset: "TransformerDataset") -> None:
        self.pos_emb.extend(sdn_dataset.pos_emb)
        self.sub_graphs_ids.extend(sdn_dataset.sub_graphs_ids)
        if self.mp >= 0:
            self.mp_emb.extend(sdn_dataset.mp_emb)

        self.labels.extend(sdn_dataset.labels)
        self.target_footprint.extend(sdn_dataset.target_footprint)
        if self.alter_labeling:
            self.altered_labels.extend(sdn_dataset.altered_labels)

        if self.max_nodes <= 0:
            self.max_nodes = sdn_dataset.max_nodes

    def _assign_data(self, data: Tuple[list[DatasetItem], list[Tensor], Tensor]) -> None:
        sample_lst, self.sub_graphs_ids, self.BOW_emb = data
        self.BOW_emb = self.BOW_emb.to(self.device)

        for idx, cell in enumerate(sample_lst):
            self.pos_emb.append(cell.pos_emb.to(self.device))
            self.mp_emb.append(None)
            self.labels.append(cell.labels.to(self.device))
            self.target_footprint.append(cell.target_footprint.to(self.device))
            self.altered_labels.append(cell.altered_labels.to(self.device))
            self.sub_graphs_ids[idx] = self.sub_graphs_ids[idx].to(self.device)

        self.max_nodes = self.pos_emb[0].shape[1]

    def process_split(self, split: list[int]) -> None:
        processed_path = os.path.join(self.path, "processed_dataset")
        data_split_path = os.path.join(processed_path, self.data_type)
        hop_path = os.path.join(data_split_path, f"hop_{self.hop}")
        labels_path = os.path.join(hop_path, "labels")
        pos_enc_path = os.path.join(hop_path, "pos_enc")
        subgraphs_path = os.path.join(hop_path, "subgraphs_ids")
        target_nodes_path = os.path.join(hop_path, "targets_nodes")
        BOW_path = os.path.join(processed_path, "BOW.pt")

        target_nodes: list[Tensor] = []

        for idx in split:
            full_labels_path = os.path.join(labels_path, f"labels_slice_{idx}.pt")
            full_pos_enc_path = os.path.join(pos_enc_path, f"pos_enc_slice_{idx}.pt")
            full_subgraphs_path = os.path.join(subgraphs_path, f"subgraph_ids_slice_{idx}.pt")
            full_target_nodes_path = os.path.join(target_nodes_path, f"target_nodes_slice_{idx}.pt")

            with open(full_labels_path, "rb") as f:
                self.labels.extend(pickle.load(f))

            with open(full_pos_enc_path, "rb") as f:
                self.pos_emb.extend(pickle.load(f))

            with open(full_subgraphs_path, "rb") as f:
                self.sub_graphs_ids.extend(pickle.load(f))

            with open(full_target_nodes_path, "rb") as f:
                target_nodes.extend(pickle.load(f))

        with open(BOW_path, "rb") as f:
            self.BOW_emb = pickle.load(f)["BOW"]

        self.BOW_emb = self.BOW_emb.to(self.device)

        self._process_each_sample(targets_nodes=target_nodes)

    def __len__(self) -> int:
        return len(self.pos_emb) if self.limit <= 0 else self.limit

    def __getitem__(self, item: int) -> DatasetItem:
        return DatasetItem(
            bow_emb=self.BOW_emb[self.sub_graphs_ids[item], :],
            pos_emb=self.pos_emb[item],
            mp_emb=None if self.mp < 0 else self.mp_emb[item],
            labels=self.labels[item],
            target_footprint=self.target_footprint[item],
            altered_labels=None if not self.alter_labeling else self.altered_labels[item],
            mask=None
        )
