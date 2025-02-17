from typing import Any, Tuple, Type, Callable, Union
import lightning as L
import torch
from torch import Tensor, nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import OneCycleLR

from dataset_handler.dataset_fetcher import dataset_split_max_size
from dataset_handler.transformer_dataset import DatasetItem
from models.BoxMLP import BoxMLP
from utils.aggregator_type import AGGREGATOR_FUNCTIONS_ds, MASKED_AGGREGATOR_FUNCTION_ss, AGGREGATOR_FUNCTIONS_ss
from utils.layer_type import LayerType
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.functional import auroc

FullBatchWrapper = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


# TODO allegedly max token size should be based on the hardware from my calculation
#  with a d_model of 1000 and a max_token of 500 3 GB of VRAM is used (without counting ff
#  and others)

class LinkTransformer(L.LightningModule):
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
            dropout: float = .0, lr: float = 0.001,
            weight_decay: float = .01,
            one_cycle_pct_start: float = 0.5,
            one_cycle_max_lr: float = 0.05,
            one_cycle_base_momentum: float = 0.85,
            one_cycle_max_momentum: float = 0.95,
            loss_fun: Type[nn.Module] = nn.BCELoss,
            alter_labeling: bool = False,
            no_pos: bool = False,
            no_trick: bool = False,
            device: str = "cuda"
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.head_agg: Callable[[Tensor, Tensor], Tensor] = MASKED_AGGREGATOR_FUNCTION_ss[head_agg_type]
        self.tail_agg: Callable[[Tensor, Tensor], Tensor] = AGGREGATOR_FUNCTIONS_ds[tail_agg_type]

        if no_pos:
            self.positional_emb: nn.Embedding = nn.Embedding(
                num_embeddings=pos_input_size, embedding_dim=pos_out_size,
                device=device
            )
        else:
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

        if no_trick:
            transformer_input_size: int = emb_out_size + pos_out_size

        else:
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
        if "aggregator" in decoder_args:
            decoder_args["aggregator"] = AGGREGATOR_FUNCTIONS_ss[decoder_args["aggregator"]]

        self.decoder = BoxMLP(**decoder_args).to(device)

        self._val_predictions: list[Tensor] = []
        self._val_targets: list[Tensor] = []

        self._test_predictions: list[Tensor] = []
        self._test_targets: list[Tensor] = []

        self.criterion: nn.Module = loss_fun()

    def no_pos_handler(self, pos_aggregated: Tensor, masks: Tensor) -> Tensor:
        if self.hparams.no_pos:
            out: Tensor = torch.zeros(
                pos_aggregated.shape[0], pos_aggregated.shape[1],
                self.hparams.pos_out_size, device=self.device
            )

            pos_aggregated = self.positional_emb(pos_aggregated)

            # out[~masks] = pos_aggregated[~masks]

            return pos_aggregated

        else:
            return self.positional_emb(pos_aggregated)

    def no_trick_handler(
            self, feats: Tensor, pos_aggregated: Tensor,
            zero_one_trick: Tensor, dist_trick: Tensor
    ) -> Tensor:
        if self.hparams.no_trick:
            aggregated = torch.cat([
                feats, pos_aggregated,
            ], dim=-1)
        else:
            if self.hparams.alter_labeling:
                chosen_trick: Tensor = dist_trick
            else:
                chosen_trick: Tensor = zero_one_trick

            aggregated = torch.cat([
                feats, pos_aggregated,
                chosen_trick.unsqueeze(dim=-1)
            ], dim=-1)

        return aggregated

    def forward(self, batch: FullBatchWrapper) -> Tensor:
        feat_batch, pos_batch, mp_batch, masks, target_footprint, altered_labels = batch

        pos_aggregated: Tensor = self.tail_agg(pos_batch, mp_batch)
        pos_aggregated = self.positional_emb(pos_aggregated)
        feats: Tensor = self.feats_emb(feat_batch)

        aggregated: Tensor = self.no_trick_handler(
            feats=feats, pos_aggregated=pos_aggregated,
            zero_one_trick=target_footprint, dist_trick=altered_labels
        )

        optimus_out: Tensor = self.transformer(aggregated, src_key_padding_mask=masks)
        optimus_out = self.head_agg(optimus_out, target_footprint)

        return self.decoder(optimus_out)

    def predict(self, batch: DatasetItem) -> Tensor:
        self.eval()

        with torch.no_grad():
            feats_batch, pos_batch, mp_emb, masks, labels, target_footprint, altered_labels = batch
            feats_batch, pos_batch, mp_emb, masks, labels, target_footprint, altered_labels = (
                feats_batch.to(self.device), pos_batch.to(self.device), mp_emb.to(self.device),
                masks.to(self.device), labels.to(self.device), target_footprint.to(self.device),
                altered_labels.to(self.device)
            )
            prediction: Tensor = self.forward(
                (feats_batch, pos_batch, mp_emb, masks, target_footprint, altered_labels)
            ).flatten()

        return prediction

    def training_step(
            self, batch: DatasetItem,
            batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        return self.base_step(batch, batch_idx, dataloader_idx, step_type="train_dataset")

    def validation_step(
            self, batch: DatasetItem,
            batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.base_step(batch, batch_idx, dataloader_idx, step_type="val_dataset")

    def test_step(
            self, batch: DatasetItem,
            batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.base_step(batch, batch_idx, dataloader_idx, step_type="test_dataset")

    def base_step(
            self, batch: DatasetItem, batch_idx: int,
            dataloader_idx: int = 0, step_type: str = None
    ) -> Tensor:
        feats_batch, pos_batch, mp_emb, masks, labels, target_footprint, altered_labels = batch
        feats_batch, pos_batch, mp_emb, masks, labels, target_footprint, altered_labels = (
            feats_batch.to(self.device), pos_batch.to(self.device), mp_emb.to(self.device),
            masks.to(self.device), labels.to(self.device), target_footprint.to(self.device),
            altered_labels.to(self.device)
        )
        prediction: Tensor = self.forward(
            (feats_batch, pos_batch, mp_emb, masks, target_footprint, altered_labels)
        ).flatten()
        loss = self.criterion(prediction, labels.float())

        if step_type == 'val_dataset':
            self._val_predictions.append(prediction)
            self._val_targets.append(labels)

        elif step_type == 'test_dataset':
            self._test_predictions.append(prediction)
            self._test_targets.append(labels)

        self.log(
            name=f"{step_type}_BCE",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def on_base_epoch_end(self, epoch_type: str) -> None:
        if epoch_type == "val":
            y_hat, y = self._val_predictions, self._val_targets
        else:
            y_hat, y = self._test_predictions, self._test_targets

        predictions = torch.concatenate(y_hat, dim=0).flatten()
        targets = torch.concatenate(y, dim=0).flatten()
        labels = (predictions >= self.hparams.likelihood_threshold).int()

        roc_auc_score: Tensor = auroc(preds=predictions, target=targets.int(), task="binary")
        accuracy_score: Tensor = torch.sum(labels == targets) / targets.shape[0]

        # I put average="macro" otherwise f1_score == accuracy_score
        f1_score: Tensor = multiclass_f1_score(
            input=labels.to(torch.int64),
            target=targets.to(torch.int64),
            num_classes=2, average="macro"
        )

        self.log_dict(
            dictionary={
                f"accuracy_{epoch_type}": accuracy_score,
                f"f1_{epoch_type}": f1_score,
                f"roc_auc_{epoch_type}": roc_auc_score
            },
            on_step=False,  # True raises error because we are on epoch end
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

    def on_validation_epoch_end(self) -> None:
        self.on_base_epoch_end(epoch_type="val")
        self._val_targets.clear()
        self._val_predictions.clear()

    def on_test_epoch_end(self) -> None:
        self.on_base_epoch_end(epoch_type="test")
        self._test_targets.clear()
        self._test_predictions.clear()

    def configure_optimizers(self) -> Tuple[list, list]:
        # Weight decay is not applied onto bias parameters and layer norm layer
        no_decay = ["bias", "layer_norm", "norm1", "norm2"]
        param_groups = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # both lr and momentum will be adjusted by OneCycleLR
        optimizer: optim.Optimizer = torch.optim.SGD(
            param_groups, lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.one_cycle_max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.one_cycle_pct_start,
            base_momentum=self.hparams.one_cycle_base_momentum,
            max_momentum=self.hparams.one_cycle_max_momentum
        )

        scheduler = {
            "name": OneCycleLR.__name__,
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return [optimizer], [scheduler]

# TODO se vedi che va tutto male togli one_cycle e metti lr a 0.01 e forse momentum
