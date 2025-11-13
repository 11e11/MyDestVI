from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, NumericalObsField, ObsmField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from src.modules.mrdeconv import MRDeconv
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Sequence

    from anndata import AnnData

    from src.models.condscvi import CondSCVI

logger = logging.getLogger(__name__)


class DestVI(UnsupervisedTrainingMixin, BaseModelClass):
    """Multi-resolution deconvolution of Spatial Transcriptomics data (DestVI) :cite:p:`Lopez22`.

    Most users will use the alternate constructor (see example).

    Parameters
    ----------
    st_adata
        spatial transcriptomics AnnData object that has been registered via
        :meth:`~scvi.model.DestVI.setup_anndata`.
    cell_type_mapping
        mapping between numerals and cell type labels
    decoder_state_dict
        state_dict from the decoder of the CondSCVI model
    px_decoder_state_dict
        state_dict from the px_decoder of the CondSCVI model
    px_r
        parameters for the px_r tensor in the CondSCVI model
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    **module_kwargs
        Keyword args for :class:`~scvi.module.MRDeconv`

    Examples
    --------
    >>> sc_adata = anndata.read_h5ad(path_to_scRNA_anndata)
    >>> scvi.model.CondSCVI.setup_anndata(sc_adata)
    >>> sc_model = scvi.model.CondSCVI(sc_adata)
    >>> st_adata = anndata.read_h5ad(path_to_ST_anndata)
    >>> DestVI.setup_anndata(st_adata)
    >>> spatial_model = DestVI.from_rna_model(st_adata, sc_model)
    >>> spatial_model.train(max_epochs=2000)
    >>> st_adata.obsm["proportions"] = spatial_model.get_proportions(st_adata)
    >>> gamma = spatial_model.get_gamma(st_adata)

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/spatial/DestVI_tutorial`
    2. :doc:`/tutorials/notebooks/r/DestVI_in_R`
    """

    _module_cls = MRDeconv

    def __init__(
        self,
        st_adata: AnnData,
        cell_type_mapping: np.ndarray,
        decoder_state_dict: OrderedDict,
        px_decoder_state_dict: OrderedDict,
        px_r: np.ndarray,
        n_hidden: int,
        n_latent: int,
        n_layers: int,
        dropout_decoder: float,
        l1_reg: float,

        dirichlet_alpha: float | list | None = None,
        dirichlet_mmd_reg: float = 0.0,

        contrastive_reg: float = 1.0,
        contrastive_margin: float = 0.5,

        **module_kwargs,
    ):
        super().__init__(st_adata)
        # 实例化MRDeconv模块
        self.module = self._module_cls(
            n_spots=st_adata.n_obs,
            n_labels=cell_type_mapping.shape[0],
            decoder_state_dict=decoder_state_dict,
            px_decoder_state_dict=px_decoder_state_dict,
            px_r=px_r,
            n_genes=st_adata.n_vars,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_decoder=dropout_decoder,
            l1_reg=l1_reg,

            dirichlet_alpha=dirichlet_alpha,
            dirichlet_mmd_reg=dirichlet_mmd_reg,

            contrastive_reg=contrastive_reg,
            contrastive_margin=contrastive_margin,

            **module_kwargs,
        )
        self.cell_type_mapping = cell_type_mapping
        self._model_summary_string = "DestVI Model"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def from_rna_model(
        cls,
        st_adata: AnnData,
        sc_model: CondSCVI,
        vamp_prior_p: int = 15,
        l1_reg: float = 0.0,
        **module_kwargs,
    ):
        """Alternate constructor for exploiting a pre-trained model on a RNA-seq dataset.
        从预训练的CondSCVI模型中提取参数，初始化DestVI模型。
        Parameters
        ----------
        st_adata
            registered anndata object
        sc_model
            trained CondSCVI model
        vamp_prior_p
            number of mixture parameter for VampPrior calculations
        l1_reg
            Scalar parameter indicating the strength of L1 regularization on cell type proportions.
            A value of 50 leads to sparser results.
        **model_kwargs
            Keyword args for :class:`~scvi.model.DestVI`
        """
        decoder_state_dict = sc_model.module.decoder.state_dict()
        px_decoder_state_dict = sc_model.module.px_decoder.state_dict()
        px_r = sc_model.module.px_r.detach().cpu().numpy()
        mapping = sc_model.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).categorical_mapping
        dropout_decoder = sc_model.module.dropout_rate
        if vamp_prior_p is None:
            mean_vprior = None
            var_vprior = None
        else:
            mean_vprior, var_vprior, mp_vprior = sc_model.get_vamp_prior(
                sc_model.adata, p=vamp_prior_p
            )

        return cls(
            st_adata,
            mapping,
            decoder_state_dict,
            px_decoder_state_dict,
            px_r,
            sc_model.module.n_hidden,
            sc_model.module.n_latent,
            sc_model.module.n_layers,
            mean_vprior=mean_vprior,
            var_vprior=var_vprior,
            mp_vprior=mp_vprior,
            dropout_decoder=dropout_decoder,
            l1_reg=l1_reg,
            **module_kwargs,
        )

    # def get_proportions(
    #     self,
    #     keep_noise: bool = False,
    #     indices: Sequence[int] | None = None,
    #     batch_size: int | None = None,
    # ) -> pd.DataFrame:
    #     """Returns the estimated cell type proportion for the spatial data.

    #     Shape is n_cells x n_labels OR n_cells x (n_labels + 1) if keep_noise.

    #     Parameters
    #     ----------
    #     keep_noise
    #         whether to account for the noise term as a standalone cell type in the proportion
    #         estimate.
    #     indices
    #         Indices of cells in adata to use. Only used if amortization. If `None`, all cells are
    #         used.
    #     batch_size
    #         Minibatch size for data loading into model. Only used if amortization. Defaults to
    #         `scvi.settings.batch_size`.
    #     """
    #     self._check_if_trained()

    #     column_names = self.cell_type_mapping
    #     index_names = self.adata.obs.index
    #     if keep_noise:
    #         column_names = np.append(column_names, "noise_term")

    #     if self.module.amortization in ["both", "proportion"]:
    #         stdl = self._make_data_loader(adata=self.adata, indices=indices, batch_size=batch_size)
    #         prop_ = []
    #         for tensors in stdl:
    #             generative_inputs = self.module._get_generative_input(tensors, None)
    #             prop_local = self.module.get_proportions(
    #                 x=generative_inputs["x"], keep_noise=keep_noise
    #             )
    #             prop_ += [prop_local.cpu()]
    #         data = torch.cat(prop_).numpy()
    #         if indices:
    #             index_names = index_names[indices]
    #     else:
    #         if indices is not None:
    #             logger.info(
    #                 "No amortization for proportions, ignoring indices and returning results for "
    #                 "the full data"
    #             )
    #         data = self.module.get_proportions(keep_noise=keep_noise)

    #     return pd.DataFrame(
    #         data=data,
    #         columns=column_names,
    #         index=index_names,
    #     )
    

    def get_proportions(
        self,
        keep_noise: bool = False,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        """Returns the estimated cell type proportion for the spatial data."""
        self._check_if_trained()

        column_names = self.cell_type_mapping
        index_names = self.adata.obs.index
        if keep_noise:
            column_names = np.append(column_names, "noise_term")

        if self.module.amortization in ["both", "proportion"]:
            # 对于 GAT，需要特殊处理
            if getattr(self.module, 'use_gat', False):
                # GAT 模式：直接获取全量结果
                data = self.module.get_proportions(keep_noise=keep_noise)
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()
                if indices is not None:
                    data = data[indices]
                    index_names = index_names[indices]
            else:
                # 原始 FC 模式：批次处理
                stdl = self._make_data_loader(adata=self.adata, indices=indices, batch_size=batch_size)
                prop_ = []
                for tensors in stdl:
                    generative_inputs = self.module._get_generative_input(tensors, None)
                    prop_local = self.module.get_proportions(
                        x=generative_inputs["x"], keep_noise=keep_noise
                    )
                    prop_ += [prop_local.cpu()]
                data = torch.cat(prop_).numpy()
                if indices:
                    index_names = index_names[indices]
        else:
            if indices is not None:
                logger.info(
                    "No amortization for proportions, ignoring indices and returning results for "
                    "the full data"
                )
            data = self.module.get_proportions(keep_noise=keep_noise)
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

        return pd.DataFrame(
            data=data,
            columns=column_names,
            index=index_names,
        )

    def get_gamma(
        self,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        return_numpy: bool = False,
    ) -> np.ndarray | dict[str, pd.DataFrame]:
        """Returns the estimated cell-type specific latent space for the spatial data.

        Parameters
        ----------
        indices
            Indices of cells in adata to use. Only used if amortization. If `None`, all cells are
            used.
        batch_size
            Minibatch size for data loading into model. Only used if amortization. Defaults to
            `scvi.settings.batch_size`.
        return_numpy
            if activated, will return a numpy array of shape is n_spots x n_latent x n_labels.
        """
        self._check_if_trained()

        column_names = np.arange(self.module.n_latent)
        index_names = self.adata.obs.index

        if self.module.amortization in ["both", "latent"]:
            stdl = self._make_data_loader(adata=self.adata, indices=indices, batch_size=batch_size)
            gamma_ = []
            for tensors in stdl:
                generative_inputs = self.module._get_generative_input(tensors, None)
                gamma_local = self.module.get_gamma(x=generative_inputs["x"])
                gamma_ += [gamma_local.cpu()]
            data = torch.cat(gamma_, dim=-1).numpy()
            if indices is not None:
                index_names = index_names[indices]
        else:
            if indices is not None:
                logger.info(
                    "No amortization for latent values, ignoring adata and returning results for "
                    "the full data"
                )
            data = self.module.get_gamma()

        data = np.transpose(data, (2, 0, 1))
        if return_numpy:
            return data
        else:
            res = {}
            for i, ct in enumerate(self.cell_type_mapping):
                res[ct] = pd.DataFrame(data=data[:, :, i], columns=column_names, index=index_names)
            return res

    def get_scale_for_ct(
        self,
        label: str,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        r"""Return the scaled parameter of the NB for every spot in queried cell types.

        Parameters
        ----------
        label
            cell type of interest
        indices
            Indices of cells in self.adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        Pandas dataframe of gene_expression
        """
        self._check_if_trained()

        if label not in self.cell_type_mapping:
            raise ValueError("Unknown cell type")
        y = np.where(label == self.cell_type_mapping)[0][0]

        stdl = self._make_data_loader(self.adata, indices=indices, batch_size=batch_size)
        scale = []
        for tensors in stdl:
            generative_inputs = self.module._get_generative_input(tensors, None)
            x, ind_x = (
                generative_inputs["x"],
                generative_inputs["ind_x"],
            )
            px_scale = self.module.get_ct_specific_expression(x, ind_x, y)
            scale += [px_scale.cpu()]

        data = torch.cat(scale).numpy()
        column_names = self.adata.var.index
        index_names = self.adata.obs.index
        if indices is not None:
            index_names = index_names[indices]
        return pd.DataFrame(data=data, columns=column_names, index=index_names)

    # @devices_dsp.dedent
    # def train(
    #     self,
    #     max_epochs: int = 2000,
    #     lr: float = 0.003,
    #     accelerator: str = "auto",
    #     devices: int | list[int] | str = "auto",
    #     train_size: float = 1.0,
    #     validation_size: float | None = None,
    #     shuffle_set_split: bool = True,
    #     batch_size: int = 128,
    #     n_epochs_kl_warmup: int = 200,
    #     datasplitter_kwargs: dict | None = None,
    #     plan_kwargs: dict | None = None,
    #     **kwargs,
    # ):
    #     """Trains the model using MAP inference.

    #     Parameters
    #     ----------
    #     max_epochs
    #         Number of epochs to train for
    #     lr
    #         Learning rate for optimization.
    #     %(param_accelerator)s
    #     %(param_devices)s
    #     train_size
    #         Size of training set in the range [0.0, 1.0].
    #     validation_size
    #         Size of the test set. If `None`, defaults to 1 - `train_size`. If
    #         `train_size + validation_size < 1`, the remaining cells belong to a test set.
    #     shuffle_set_split
    #         Whether to shuffle indices before splitting. If `False`, the val, train, and test set
    #         are split in the sequential order of the data according to `validation_size` and
    #         `train_size` percentages.
    #     batch_size
    #         Minibatch size to use during training.
    #     n_epochs_kl_warmup
    #         number of epochs needed to reach unit kl weight in the elbo
    #     datasplitter_kwargs
    #         Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
    #     plan_kwargs
    #         Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
    #         `train()` will overwrite values present in `plan_kwargs`, when appropriate.
    #     **kwargs
    #         Other keyword args for :class:`~scvi.train.Trainer`.
    #     """
    #     update_dict = {
    #         "lr": lr,
    #         "n_epochs_kl_warmup": n_epochs_kl_warmup,
    #     }
    #     if plan_kwargs is not None:
    #         plan_kwargs.update(update_dict)
    #     else:
    #         plan_kwargs = update_dict
    #     super().train(
    #         max_epochs=max_epochs,
    #         accelerator=accelerator,
    #         devices=devices,
    #         train_size=train_size,
    #         validation_size=validation_size,
    #         shuffle_set_split=shuffle_set_split,
    #         batch_size=batch_size,
    #         datasplitter_kwargs=datasplitter_kwargs,
    #         plan_kwargs=plan_kwargs,
    #         **kwargs,
    #     )

    def train(self, max_epochs: int = 2000, **kwargs):
        """支持对比学习：仅当 padded 索引存在时启用"""
        use_contrastive = "pos_indices_padded" in self.adata_manager.adata.obsm

        # 清理无关透传参数
        kwargs.pop("data_loader_class", None)
        kwargs.pop("datasplitter_kwargs", None)

        if use_contrastive:
            print("🎯 Using contrastive learning!")
            try:
                from src.utils.ContrasiveDataLoader import ContrastiveAnnDataLoader
                self._data_loader_cls = ContrastiveAnnDataLoader

                if hasattr(self.module, "attach_full_X"):
                    self.module.attach_full_X(self.adata_manager.adata)

                plan_kwargs = (kwargs.get("plan_kwargs") or {}).copy()
                plan_kwargs.update({
                    "lr": kwargs.get("lr", 0.003),
                    "n_epochs_kl_warmup": kwargs.get("n_epochs_kl_warmup", 200),
                })
                kwargs["plan_kwargs"] = plan_kwargs

                print(f"Starting contrastive training with {self._data_loader_cls.__name__} ...")
                return super().train(max_epochs=max_epochs, **kwargs)

            except Exception as e:
                print(f"Contrastive training failed: {e}")
                print("Falling back to standard training...")
                return self._standard_train(max_epochs, **kwargs)
        else:
            print("📊 Using standard training (no contrastive pairs)")
            return self._standard_train(max_epochs, **kwargs)



    def _standard_train(self, max_epochs: int = 2000, **kwargs):
        """标准训练方法"""
        plan_kwargs = kwargs.get('plan_kwargs', {})
        plan_kwargs.update({
            'lr': kwargs.get('lr', 0.003),
            'n_epochs_kl_warmup': kwargs.get('n_epochs_kl_warmup', 200)
        })
        kwargs['plan_kwargs'] = plan_kwargs
        return super().train(max_epochs=max_epochs, **kwargs)

    # @classmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     cls,
    #     adata: AnnData,
    #     layer: str | None = None,
    #     **kwargs,
    # ):
    #     """%(summary)s.

    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     %(param_layer)s
    #     """
    #     setup_method_args = cls._get_setup_method_args(**locals())
    #     # add index for each cell (provided to pyro plate for correct minibatching)
    #     adata.obs["_indices"] = np.arange(adata.n_obs)
    #     anndata_fields = [
    #         LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
    #         NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
    #     ]
    #     # 只有当正负对数据存在时才添加对应字段
    #     if "pos_indices" in adata.obsm:
    #         anndata_fields.extend([
    #             ObsmField("pos_indices", "pos_indices"),
    #             ObsmField("neg_indices", "neg_indices"),
    #         ])
        
    #     if "pos_indices_count" in adata.obs:
    #         anndata_fields.extend([
    #             NumericalObsField("pos_indices_count", "pos_indices_count"),
    #             NumericalObsField("neg_indices_count", "neg_indices_count"),
    #         ])
    #     adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
    #     adata_manager.register_fields(adata, **kwargs)
    #     cls.register_manager(adata_manager)
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata,
        layer: str | None = None,
        **kwargs,
    ):
        """
        注册 DestVI 所需的 AnnData 字段。
        - X: 来自指定 layer（计数矩阵）
        - _indices: 每个观测的整数索引
        - pos_indices_padded/neg_indices_padded: 填充后的二维索引矩阵（在 obsm）
        - pos_indices_count/neg_indices_count: 每行有效数量（在 obs）
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        # 为每个观测创建索引列，供 DataLoader 使用
        adata.obs["_indices"] = np.arange(adata.n_obs, dtype=int)

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]

        # 仅注册 padded 字段，拒绝 ragged
        if "pos_indices_padded" in adata.obsm and "neg_indices_padded" in adata.obsm:
            anndata_fields.extend([
                ObsmField("pos_indices_padded", "pos_indices_padded"),
                ObsmField("neg_indices_padded", "neg_indices_padded"),
            ])
        elif "pos_indices" in adata.obsm or "neg_indices" in adata.obsm:
            raise ValueError("检测到 ragged 的 pos/neg_indices。请先调用 retrofit_padding_for_contrastive_pairs(adata)。")

        if "pos_indices_count" in adata.obs and "neg_indices_count" in adata.obs:
            anndata_fields.extend([
                NumericalObsField("pos_indices_count", "pos_indices_count"),
                NumericalObsField("neg_indices_count", "neg_indices_count"),
            ])

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)