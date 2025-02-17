from typing import Type, Callable
from torch import nn, Tensor
from utils.MLPBundle import MLPBundle
from models.base.superMLP import BaseSuperMLP
from utils.layer_type import LayerType


class BoxMLP(BaseSuperMLP):
    def __init__(
            self, n_layers: int, in_features: int, out_features: int,
            aggregator: Callable[[Tensor], Tensor] = None,
            f: Type[nn.Module] = nn.Sigmoid, alpha: float = 0.5,
            dropout: float = .2, linear: bool = False, layernorm: bool = False,
            mid_out: int = None, tensor_dot: bool = True,
            block_type: LayerType = LayerType.SUPER_BLOCK,
            num_target: int = 2, merge: bool = True,
            no_head: bool = False
    ) -> None:

        layers_spec: list[MLPBundle] = []

        if linear:
            layers_spec = self._linear_setup(
                n_layers=n_layers, in_features=in_features,
                out_features=out_features, layernorm=layernorm,
                dropout=dropout, aggregator=aggregator,
                num_target=num_target, merge=merge
            )
        else:
            layers_spec = self._block_setup(
                block_type=block_type, n_layers=n_layers, in_features=in_features,
                out_features=out_features, f=f, layernorm=layernorm, alpha=alpha,
                dropout=dropout, mid_out=mid_out, aggregator=aggregator,
                tensor_dot=tensor_dot, num_target=num_target, merge=merge,
                no_head=no_head
            )

        self.tensor_dot: bool = tensor_dot

        super().__init__(layers_spec, f)

    def _block_setup(
            self, block_type: LayerType, n_layers: int, in_features: int,
            out_features: int, f: Type[nn.Module], layernorm: bool,
            aggregator: Callable[[Tensor], Tensor],
            dropout: float, alpha: float = None, num_target: int = 2,
            tensor_dot: bool = True, mid_out: int = None, merge: bool = True,
            no_head: bool = False
    ) -> list[MLPBundle]:

        layers_spec: list[MLPBundle] = []
        mid_out = mid_out if mid_out else in_features

        bounds: tuple[int, int] = (1, n_layers - 1) if no_head else (0, n_layers - 1)

        if no_head:
            layers_spec += self.set_a_block(
                in_features=in_features, out_features=in_features,
                idx=0, f=f, layernorm=layernorm, dropout=dropout,
                n_layers=n_layers, block_type=LayerType.DECOMPOSER,
                tensor_dot=tensor_dot, num_target=num_target
            )

        for idx in range(*bounds):
            layers_spec += self.set_a_block(
                in_features=in_features, mid_out=mid_out, out_features=in_features,
                idx=idx, f=f, layernorm=layernorm, dropout=dropout,
                n_layers=n_layers, block_type=block_type, aggregator=aggregator,
                tensor_dot=tensor_dot, num_target=num_target
            )

        if merge and num_target > 1:
            layers_spec += self.set_a_block(
                in_features=in_features, mid_out=mid_out, out_features=out_features,
                idx=n_layers - 1, layernorm=False, n_layers=n_layers, block_type=LayerType.AGGREGATOR,
                aggregator=aggregator, tensor_dot=tensor_dot, num_target=1, f=f, dropout=dropout
            )
        else:
            layers_spec += self.set_a_block(
                in_features=in_features, mid_out=mid_out, out_features=out_features,
                idx=n_layers - 1, f=f, layernorm=layernorm, dropout=dropout,
                n_layers=n_layers, block_type=block_type, aggregator=None,
                tensor_dot=tensor_dot, num_target=num_target
            )

        return layers_spec

    def _linear_setup(
            self, n_layers: int, in_features: int, out_features: int,
            layernorm: bool, dropout: float, aggregator: Callable[[Tensor], Tensor],
            num_target: int = 2, merge: bool = True
    ) -> list[MLPBundle]:

        layers_spec: list[MLPBundle] = []

        for idx in range(n_layers - 1):
            layers_spec += self.set_linear_layer(
                in_features=in_features, out_features=in_features,
                idx=idx, layernorm=layernorm, dropout=dropout, n_layers=n_layers,
                num_target=num_target
            )

        if merge and num_target > 1:
            layers_spec += self.set_a_block(
                in_features=in_features, mid_out=in_features, out_features=out_features,
                idx=n_layers - 1, layernorm=False, n_layers=n_layers, block_type=LayerType.AGGREGATOR,
                aggregator=aggregator, tensor_dot=False, num_target=1, f=nn.Sigmoid, dropout=dropout,
            )
        else:
            layers_spec += self.set_linear_layer(
                in_features=in_features, out_features=out_features,
                idx=n_layers - 1, layernorm=layernorm, dropout=dropout,
                n_layers=n_layers, num_target=num_target
            )

        return layers_spec

    def _dec_forbs_agg(self, n_layers):
        pass
