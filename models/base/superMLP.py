from typing import Type, Callable, Tuple, Any
from torch import nn, Tensor
from utils.MLPBundle import MLPBundle
from utils.layer_type import LayerType


class BaseSuperMLP(nn.Module):
    def __init__(
            self, layers_info: list[MLPBundle], f: Type[nn.Module],
    ) -> None:

        super().__init__()

        self._n_layers: int = len(layers_info)
        self.model_layers, self.model_layers_names = self._build_model(layers_info, f)

    def _build_model(self, layers_info: list[MLPBundle], f: Type[nn.Module]) -> Tuple[nn.ModuleList, list[str]]:
        model_layers: nn.ModuleList = nn.ModuleList()
        model_layers_names: list[str] = []
        f_counter: int = 1

        for idx, layer_info in enumerate(layers_info):
            layer_type: Type[nn.Module] = layer_info.get_type()
            args: dict[str, Any] = layer_info.get_bundle()
            model_layers.append(layer_type(**args))
            model_layers_names.append(layer_info.name)

            flag: bool = True
            if self._n_layers - 1 > idx:
                nxt_layer_type: Type[nn.Module] = layers_info[idx + 1].get_type()
                flag = nxt_layer_type != nn.LayerNorm

            if layer_type != nn.Dropout and flag:
                model_layers.append(f())
                model_layers_names.append(f"f_{f_counter}")
                f_counter += 1

        return model_layers, model_layers_names

    def forward(self, batch: Tensor) -> Tensor:
        x_batch = batch
        for layer_name, layer in zip(self.model_layers_names, self.model_layers):
            x_batch = layer(x_batch)

        return x_batch

    def set_a_block(
            self, block_type: LayerType, in_features: int, out_features: int,
            idx: int, f: Type[nn.Module], layernorm: bool, dropout: float,
            n_layers: int, mid_out: int = None, alpha: float = None,
            aggregator: Callable[[Tensor], Tensor] = None, tensor_dot: bool = True,
            num_target: int = 2
    ) -> list[MLPBundle]:

        layer_spec: list[MLPBundle] = []

        if (not idx) or (idx == n_layers - 1):
            dropout = .0

        bundle: MLPBundle = MLPBundle(
            index=idx + 1, in_features=in_features, mid_out=mid_out,
            out_features=out_features, alpha=alpha, dropout=dropout,
            l_type=block_type, f=f, aggregator=aggregator,
            tensor_dot=tensor_dot, num_target=num_target,
        )

        layer_spec.append(bundle)

        if layernorm and idx != n_layers - 1:
            layer_spec.append(
                MLPBundle(
                    index=idx + 1, emb_size=out_features,
                    l_type=LayerType.LAYER_NORM
                )
            )

        return layer_spec

    def set_linear_layer(
            self, in_features: int, out_features: int, idx: int,
            layernorm: bool, dropout: float, n_layers: int,
            num_target: int = 2
    ) -> list[MLPBundle]:

        layer_spec: list[MLPBundle] = []

        if idx >= n_layers - 1:
            dropout = .0

        layer_spec.append(MLPBundle(
            index=idx + 1, in_features=in_features,
            out_features=out_features, l_type=LayerType.LINEAR,
            num_target=num_target
        ))

        if layernorm and idx != n_layers - 1:
            layer_spec.append(
                MLPBundle(
                    index=idx + 1, emb_size=out_features,
                    l_type=LayerType.LAYER_NORM)
            )

        if dropout > .0:
            layer_spec.append(
                MLPBundle(index=idx + 1, dropout=dropout, l_type=LayerType.DROPOUT)
            )

        return layer_spec
