from typing import Any, Union, Callable, Tuple, Type

import torch
from torch import nn, Tensor, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.functional import auroc

from dataset_handler.transformer_dataset import DatasetItem
from models.BoxMLP import BoxMLP
from utils.aggregator_type import MASKED_AGGREGATOR_FUNCTION_ss, AGGREGATOR_FUNCTIONS_ds
from utils.layer_type import LayerType

FullBatchWrapper = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def early_stop_handler(idx: int, val_loss: list[float], min_delta: float) -> bool:
    return (val_loss[idx] - val_loss[-1]) < min_delta


class LinkTransformer(nn.Module):
    def __init__(
            self, transformer_n_layer: int, n_attention_head: int,
            emb_input_size: int, pos_input_size: int,
            pos_out_size: int, emb_out_size: int,
            tail_agg_type: str, head_agg_type: str,
            decoder_args: dict[str, Any],
            encoder_activation: Union[str, Callable[[Tensor], Tensor]] = "gelu",
            feedforward_dim: int = 2048,
            norm_first: bool = False, bias: bool = True,
            likelihood_threshold: float = 0.5,
            dropout: float = .0, lr: float = 0.01,
            weight_decay: float = .0,
            loss_fun: Type[nn.Module] = nn.BCELoss,
            device: str = "cuda"
    ) -> None:
        super().__init__()

        self.head_agg: Callable[[Tensor, Tensor], Tensor] = MASKED_AGGREGATOR_FUNCTION_ss[head_agg_type]
        self.tail_agg: Callable[[Tensor, Tensor], Tensor] = AGGREGATOR_FUNCTIONS_ds[tail_agg_type]

        self.positional_emb: nn.Linear = nn.Linear(
            in_features=pos_input_size,
            out_features=pos_out_size,
            device=device
        )

        self.feats_emb: nn.Linear = nn.Linear(
            in_features=emb_input_size,
            out_features=emb_out_size,
            device=device
        )

        transformer_input_size: int = emb_out_size + pos_out_size + 1

        self.transformer: TransformerEncoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=transformer_input_size, nhead=n_attention_head,
                dim_feedforward=feedforward_dim, activation=encoder_activation,
                norm_first=norm_first, bias=bias,
                device=device, batch_first=True,
                dropout=dropout
            ),
            num_layers=transformer_n_layer,
        )

        decoder_args["block_type"] = LayerType[decoder_args["block_type"]]
        self.decoder = nn.Linear(
            in_features=800,
            out_features=1,
            device=device
        )  # BoxMLP(**decoder_args).to(device)

        self.likelihood_threshold: float = likelihood_threshold
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.optimizer: optim.Optimizer = self.get_optimizer()
        self.criterion: nn.Module = loss_fun()
        self.device: str = device

        self.best_hyper: dict[str, Any] = {
            "val_loss": -1,
            "accuracy": -1,
            "roc_auc": -1,
            "f1": -1,
            "epoch": -1
        }

    def forward(self, batch: FullBatchWrapper) -> Tensor:
        feats_batch, pos_batch, mp_emb, masks, target_footprint = batch
        feats_batch, pos_batch, mp_emb, masks, target_footprint = (
            feats_batch.to(self.device), pos_batch.to(self.device),
            mp_emb.to(self.device), masks.to(self.device),
            target_footprint.to(self.device)
        )

        pos_aggregated: Tensor = self.tail_agg(pos_batch, mp_emb)
        pos_aggregated = self.positional_emb(pos_aggregated)
        feats: Tensor = self.feats_emb(feats_batch)

        aggregated = torch.cat([
            feats, pos_aggregated,
            target_footprint.unsqueeze(dim=-1)
        ], dim=-1)

        # TODO see also if you need to apply a non-linearity
        optimus_out: Tensor = self.transformer(aggregated, src_key_padding_mask=masks)

        optimus_out = self.head_agg(optimus_out, target_footprint)

        return nn.Sigmoid()(self.decoder(optimus_out))

    def get_optimizer(self) -> optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]  # Parameters to exclude from weight decay
        param_groups = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.SGD(
            param_groups, lr=self.lr,
            weight_decay=self.weight_decay
        )

    def base_step(self, batch: DatasetItem) -> Tuple[Tensor, Tensor]:
        feats_batch, pos_batch, mp_emb, masks, labels, target_footprint = batch
        prediction: Tensor = self.forward((
            feats_batch, pos_batch, mp_emb, masks, target_footprint
        )).flatten()

        return prediction, self.criterion(prediction, labels.to(self.device).float())

    def update_best_hyper(
            self, val_loss: float, accuracy: Tensor,
            roc_auc: Tensor, f1: Tensor, epoch: int
    ) -> None:

        if self.best_hyper["val_loss"] < 0 or self.best_hyper["val_loss"] > val_loss:
            self.best_hyper = {
                "val_loss": val_loss,
                "accuracy": accuracy.item(),
                "roc_auc": roc_auc.item(),
                "f1": f1.item(),
                "epoch": epoch
            }

    def validation_handler(self, val_data: DataLoader, epoch: int, log: int) -> float:
        self.eval()

        predictions: Tensor = torch.tensor([], device=self.device)
        targets: Tensor = torch.tensor([], device=self.device)
        total_loss: float = .0

        with torch.no_grad():
            for batch in val_data:
                prediction, loss = self.base_step(batch=batch)

                total_loss += loss.item()
                predictions = torch.cat((predictions, prediction), dim=0)
                targets = torch.cat((targets, batch[4].to(self.device)), dim=0)

            labels = (predictions >= self.likelihood_threshold).int()
            roc_auc_score: Tensor = auroc(preds=predictions, target=targets.int(), task="binary")
            accuracy_score: Tensor = torch.sum(labels == targets) / targets.shape[0]
            f1_score: Tensor = multiclass_f1_score(input=labels, target=targets, num_classes=2)

        if log:
            print(f"EPOCH: {epoch + 1}, ACCURACY: {accuracy_score}, ROC_AUC: {roc_auc_score}, F1: {f1_score}")

        self.update_best_hyper(
            val_loss=total_loss, accuracy=accuracy_score,
            roc_auc=roc_auc_score, f1=f1_score, epoch=epoch
        )

        self.train()

        return total_loss

    def train_handler(
            self, epochs: int, train_data: DataLoader,
            val_data: DataLoader, patience: int = -1,
            min_delta: float = .0, log: bool = True
    ):
        self.train()

        check_point: int = 0
        val_loss: list[float] = []

        if log:
            print("Starting training")

        for epoch in range(epochs):
            epoch_loss: float = .0
            for batch in train_data:
                loss: Tensor = self.base_step(batch=batch)[1]
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if log:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Epoch_Loss: {epoch_loss:.4f}, "
                    f"avg_loss: {epoch_loss / len(train_data):.4f}"
                )

            val_loss.append(self.validation_handler(val_data=val_data, epoch=epoch, log=log))

            if patience > 0 and epoch > 0 and (epoch % patience) == 0:
                if early_stop_handler(idx=check_point, val_loss=val_loss, min_delta=min_delta):
                    print(
                        f"Early stopping terminated the training, "
                        f"here listed the best_result: ", self.best_hyper
                    )
                    break
                else:
                    check_point = epoch
