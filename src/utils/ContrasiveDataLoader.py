# 只清理脏参数，其他交给 AnnDataLoader（它会输出已注册的 X/ind_x/pos_indices/neg_indices 和 *_count）
from scvi.dataloaders import AnnDataLoader

class ContrastiveAnnDataLoader(AnnDataLoader):
    def __init__(
        self,
        adata_manager,
        indices=None,
        batch_size: int = 128,
        shuffle: bool = False,
        drop_last: bool = False,
        load_sparse_tensor: bool = False,
        pin_memory: bool = True,
        **kwargs,
    ):
        # Lightning / DataSplitter 可能塞进来的键，必须丢弃（否则会透传到 torch DataLoader 报错）
        kwargs.pop("data_loader_kwargs", None)
        kwargs.pop("data_and_attributes", None)

        super().__init__(
            adata_manager,
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            load_sparse_tensor=load_sparse_tensor,
            pin_memory=pin_memory,
            **kwargs,
        )