import os
from typing import Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.functional import auroc
from dataset_handler.data_loader_fetcher import transformer_prepare_data_loaders
import models.link_transformer.encoder_decoder as ed
from dataset_handler.dataset_fetcher import DatasetType

path_lst_CORA: list[str] = [
    "version_17", "version_18_(3ly_tran)", "version_20",
    "version_22_altered", "version_23_mp_1_altered",
    "version_26_FORBIDDEN_altered",
    "version_27_3_ly_altered"
]
epoch_lst_CORA: list[str] = [
    "epoch=34.ckpt", "epoch=34.ckpt", "epoch=17.ckpt",
    "epoch=29.ckpt", "epoch=21.ckpt", "epoch=22.ckpt",
    "epoch=18.ckpt"
]

path_lst_CITESEER: list[str] = [
    "version_42_altered"
]
epoch_lst_CITESEER: list[str] = [
    "epoch=12.ckpt"
]


def eval_mrr(y_pred_pos: Tensor, y_pred_neg: Tensor):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits1_list = (ranking_list <= 1).to(torch.float)
    hits5_list = (ranking_list <= 5).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)

    return {'hits@1_list': hits1_list.mean(),
            'hits@5_list': hits5_list.mean(),
            'hits@20_list': hits20_list.mean(),
            'hits@50_list': hits50_list.mean(),
            'hits@10_list': hits10_list.mean(),
            'hits@100_list': hits100_list.mean(),
            'mrr_list': mrr_list.mean()}


def metric_handler(
        base_path: str, dataset_name: DatasetType, model_version: str, epoch_name: str, mp: bool,
        altered: bool, hit_k_datasets: list[DataLoader] = None
) -> Tensor:
    model_path: str = os.path.join(base_path, f"{dataset_name.name}_logs\\{model_version}")
    checkpoint_path: str = os.path.join(model_path, f"checkpoints\\{epoch_name}")
    hparams_file: str = os.path.join(model_path, "hparams.yaml")
    target: Tensor

    print("Model path: ", model_path)

    model: ed.LinkTransformer = ed.LinkTransformer.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file
    )

    if dataset_name == DatasetType.PUBMED:
        _, test = transformer_prepare_data_loaders(
            hop=2, mp=-1 if not mp else 1, path=os.path.join(base_path, dataset_name.name), device="cpu", batch_size=32,
            num_workers=5, prepare_test=True, alter_labeling=altered, test_shuffle=False, test_split=[x for x in range(7)],
            just_test_or_val=True, val_split=[0, 1, 2, 3, 4]
        )
    else:
        _, test = transformer_prepare_data_loaders(
            hop=2, mp=-1 if not mp else 1, path=os.path.join(base_path, dataset_name.name), device="cpu", batch_size=32,
            num_workers=5, prepare_test=True, alter_labeling=altered, test_shuffle=False,
            just_test_or_val=True
        )

    preds = []
    targets = []

    for batch in test:
        prediction = model.predict(batch=batch)
        preds.append(prediction)

        targets.append(batch.labels)

    predictions = torch.cat(preds).cuda()
    target = torch.cat(targets).cuda()

    roc_auc_score: Tensor = auroc(preds=predictions, target=target.int(), task="binary")
    print(f"roc_auc: {roc_auc_score}")

    # TODO
    """if hit_k_datasets is not None: 
        eval_mrr(
            model=model,
            datasets=hit_k_datasets,
            target_scores=predictions
        )
    """
    for thr in np.arange(0.1, 0.95, 0.05):
        labels = (predictions >= thr).int()
        accuracy_score: Tensor = torch.sum(labels == target) / target.shape[0]
        f1_score: Tensor = multiclass_f1_score(
            input=labels.to(torch.int64),
            target=target.to(torch.int64),
            num_classes=2, average="macro"
        )

        print(f"Threshold: {thr}, accuracy: {accuracy_score}, f1: {f1_score}")

    return predictions


# Use-case example
"""
    dataset_name: str = `CORA`
    dataset_path: str = os.path.join(os.getcwd(), dataset_name)
    for model_name, epoch_name in zip(path_lst_CORA, epoch_lst_CORA):
        mp: bool = True if "mp_1" in model_name else False
        altered: bool = True if "altered" in model_name else False
        
        metric_handler(
            dataset_path=dataset_path, model_version=model_name, 
            epoch_name=epoch_name, mp=mp, altered=altered
        )
"""
