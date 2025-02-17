from typing import Callable
import optuna
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch import Tensor
from dataset_handler.data_loader_fetcher import transformer_prepare_data_loaders
from dataset_handler.dataset_fetcher import dataset_sizes, DatasetType, dataset_split_max_size
from dataset_handler.poor_man_dataset import CustomDataModule
from models.link_transformer.encoder_decoder import LinkTransformer
from utils.aggregator_type import AggregatorType
from utils.layer_type import LayerType


def just_a_wrapper(path: str, dataset_name: DatasetType) -> Callable[[optuna.trial.Trial], Tensor]:
    def link_transformer_objective(trial):

        """Dataset params"""
        hop: int = trial.suggest_categorical('hop', [1, 2])

        tail_pos_in: int = dataset_split_max_size[dataset_name.value][hop - 1]
        tail_feat_in: int = dataset_sizes[dataset_name.value][-1]

        """Tail params"""
        tail_feat_out: int = trial.suggest_categorical('tail_feat_out', [600, 700, 800])
        tail_pos_out: int = trial.suggest_categorical('tail_pos_out', [99, 199])

        """Encoder params"""
        encoder_n_layer: int = trial.suggest_categorical('encoder_n_layer', [1, 2, 3, 4])
        encoder_n_head: int = trial.suggest_categorical('encoder_n_head', [2, 4, 10])
        encoder_act: str = trial.suggest_categorical('encoder_activation', ["relu", "gelu"])
        encoder_ff_dim: int = trial.suggest_categorical('encoder_ff_dim', [1400, 1600])
        encoder_dropout: int = trial.suggest_categorical('encoder_dropout', [.0, 0.1, 0.2, 0.3])

        """Head aggregator params"""
        head_emb_agg: str = trial.suggest_categorical(
            'head_emb_agg',
            [
                AggregatorType.SUM.name,
                AggregatorType.AVERAGE.name,
                AggregatorType.HADAMARD.name
            ]
        )

        """Decoder params"""
        decoder_n_layer: int = trial.suggest_categorical('decoder_n_layer', [1, 2, 3, 4])
        decoder_block_type: str = LayerType.LINEAR.name
        decoder_dropout: float = trial.suggest_categorical('decoder_dropout', [.0, 0.1, 0.2])

        if decoder_n_layer > 1:
            decoder_layer_norm: bool = trial.suggest_categorical('decoder_layer_norm', [True, False])
        else:
            decoder_layer_norm: bool = False

        """Scheduler params"""
        one_cycle_max_lr: float = trial.suggest_loguniform(name="one_cycle_max_lr", low=8e-5, high=1e-2)
        one_cycle_pct_start: float = trial.suggest_loguniform(name="one_cycle_pct_start", low=0.05, high=0.3)
        weight_decay: float = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

        alter_labeling: bool = trial.suggest_categorical('alter_labeling', [True, False])

        model = LinkTransformer(
            tail_agg_type=AggregatorType.NONE.name,
            transformer_n_layer=encoder_n_layer,
            n_attention_head=encoder_n_head,
            emb_input_size=tail_feat_in,
            pos_input_size=tail_pos_in,
            feedforward_dim=encoder_ff_dim,
            pos_out_size=tail_pos_out,
            emb_out_size=tail_feat_out,
            dropout=encoder_dropout,
            encoder_activation=encoder_act,
            head_agg_type=head_emb_agg,
            decoder_args={
                "n_layers": decoder_n_layer,
                "in_features": tail_pos_out + tail_feat_out + 1,
                "out_features": 1,
                "block_type": decoder_block_type,
                "dropout": decoder_dropout,
                "layernorm": decoder_layer_norm,
                "linear": True if decoder_block_type == LayerType.LINEAR.name else False,
                "num_target": 1
            },
            one_cycle_max_lr=one_cycle_max_lr,
            one_cycle_pct_start=one_cycle_pct_start,
            weight_decay=weight_decay,
            alter_labeling=alter_labeling
        )

        trainer = Trainer(
            max_epochs=10,
            logger=True,
            check_val_every_n_epoch=5,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[EarlyStopping(monitor="accuracy_val", patience=5, verbose=False, mode="max", min_delta=0.0)]
        )

        trainer.fit(model=model, datamodule=CustomDataModule(
            dataset_path=path, hop=hop,
            max_nodes=dataset_split_max_size[dataset_name.value][1],

        ))

        bce: Tensor = trainer.callback_metrics["val_dataset_BCE"]

        return bce

    return link_transformer_objective


def just_a_wrapper2(path: str, dataset_name: DatasetType) -> Callable[[optuna.trial.Trial], Tensor]:
    def link_transformer_objective(trial):

        """Dataset params"""
        hop: int = trial.suggest_categorical('hop', [1, 2])

        # TODO change to dataset_split_max_size[dataset_name.value][hop - 1]
        tail_pos_in: int = dataset_sizes[dataset_name.value][0]
        tail_feat_in: int = dataset_sizes[dataset_name.value][-1]

        """Tail params"""
        tail_feat_out: int = trial.suggest_categorical('tail_feat_out', [600, 700, 800])
        tail_pos_out: int = trial.suggest_categorical('tail_pos_out', [99, 199])

        """Encoder params"""
        encoder_n_layer: int = trial.suggest_categorical('encoder_n_layer', [1, 2, 3, 4])
        encoder_n_head: int = trial.suggest_categorical('encoder_n_head', [2, 4, 10])
        encoder_act: str = trial.suggest_categorical('encoder_activation', ["relu", "gelu"])
        encoder_ff_dim: int = trial.suggest_categorical('encoder_ff_dim', [1400, 1600])
        encoder_dropout: int = trial.suggest_categorical('encoder_dropout', [.0, 0.1, 0.2, 0.3])

        """Head aggregator params"""
        head_emb_agg: str = trial.suggest_categorical(
            'head_emb_agg',
            [
                AggregatorType.SUM.name,
                AggregatorType.AVERAGE.name,
                AggregatorType.HADAMARD.name
            ]
        )

        """Decoder params"""
        decoder_n_layer: int = trial.suggest_categorical('decoder_n_layer', [1, 2, 3, 4])
        decoder_block_type: str = LayerType.LINEAR.name
        decoder_dropout: float = trial.suggest_categorical('decoder_dropout', [.0, 0.1, 0.2])

        if decoder_n_layer > 1:
            decoder_layer_norm: bool = trial.suggest_categorical('decoder_layer_norm', [True, False])
        else:
            decoder_layer_norm: bool = False

        """Scheduler params"""
        one_cycle_max_lr: float = trial.suggest_loguniform(name="one_cycle_max_lr", low=8e-5, high=1e-2)
        one_cycle_pct_start: float = trial.suggest_loguniform(name="one_cycle_pct_start", low=0.05, high=0.3)
        weight_decay: float = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

        alter_labeling: bool = trial.suggest_categorical('alter_labeling', [True, False])

        model = LinkTransformer(
            tail_agg_type=AggregatorType.NONE.name,
            transformer_n_layer=encoder_n_layer,
            n_attention_head=encoder_n_head,
            emb_input_size=tail_feat_in,
            pos_input_size=tail_pos_in,
            feedforward_dim=encoder_ff_dim,
            pos_out_size=tail_pos_out,
            emb_out_size=tail_feat_out,
            dropout=encoder_dropout,
            encoder_activation=encoder_act,
            head_agg_type=head_emb_agg,
            decoder_args={
                "n_layers": decoder_n_layer,
                "in_features": tail_pos_out + tail_feat_out + 1,
                "out_features": 1,
                "block_type": decoder_block_type,
                "dropout": decoder_dropout,
                "layernorm": decoder_layer_norm,
                "linear": True if decoder_block_type == LayerType.LINEAR.name else False,
                "num_target": 1
            },
            one_cycle_max_lr=one_cycle_max_lr,
            one_cycle_pct_start=one_cycle_pct_start,
            weight_decay=weight_decay,
            alter_labeling=alter_labeling,
            no_pos=True
        )

        trainer = Trainer(
            max_epochs=6,
            logger=True,
            callbacks=[EarlyStopping(monitor="val_dataset_BCE", patience=5, verbose=False, mode="min", min_delta=0.0)],
        )

        train_dataset, val_dataset = transformer_prepare_data_loaders(
            hop=hop, mp=-1, path=path, device="cpu", batch_size=32,
            num_workers=4, max_nodes=dataset_split_max_size[dataset_name.value][hop - 1],
            alter_labeling=alter_labeling, no_pos=True, local_pos=False
        )

        trainer.fit(model=model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)

        bce: Tensor = trainer.callback_metrics["val_dataset_BCE"]

        return bce

    return link_transformer_objective
