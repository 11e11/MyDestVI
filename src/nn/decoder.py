"""解码器组件"""
import torch
import torch.nn as nn
from .layers import FCLayers


class Decoder(nn.Module):
    """解码器"""
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.05,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        inject_covariates: bool = True
    ):
        super().__init__()
        
        self.decoder_network = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=inject_covariates
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softplus()
        )
        
    def forward(self, z: torch.Tensor, cat_list: list[torch.Tensor] | None = None):
        h = self.decoder_network(z, cat_list)
        return self.output_layer(h)