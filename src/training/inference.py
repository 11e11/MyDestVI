"""推理引擎"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


class InferenceEngine:
    """推理引擎"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def get_proportions(self, dataloader, keep_noise=False):
        """获取细胞类型比例"""
        all_props = []
        
        for batch in dataloader:
            x = batch['X'].to(self.device)
            
            if hasattr(self.model, 'use_gat') and self.model.use_gat:
                # GAT模式：全图推理
                props = self.model.get_proportions(keep_noise=keep_noise)
                if isinstance(props, torch.Tensor):
                    props = props.cpu().numpy()
                # 提取当前batch的索引
                ind_x = batch['ind_x'].cpu().numpy()
                all_props.append(props[ind_x])
            else:
                # 普通模式：批次推理
                props = self.model.get_proportions(x, keep_noise)
                if isinstance(props, torch.Tensor):
                    props = props.cpu().numpy()
                all_props.append(props)
        
        return np.vstack(all_props)
    
    @torch.no_grad()
    def get_gamma(self, dataloader):
        """获取gamma参数"""
        all_gamma = []
        
        for batch in dataloader:
            x = batch['X'].to(self.device)
            
            if hasattr(self.model, 'use_gat') and self.model.use_gat:
                gamma = self.model.get_gamma()
                ind_x = batch['ind_x'].cpu().numpy()
                # gamma shape: [n_latent, n_labels, n_spots]
                all_gamma.append(gamma[:, :, ind_x])
            else:
                gamma = self.model.get_gamma(x)
                all_gamma.append(gamma)
        
        return np.concatenate(all_gamma, axis=-1)
    
    @torch.no_grad()
    def get_latent_representation(self, dataloader, give_mean=True):
        """获取潜在表示（用于CondSCVI）"""
        all_z = []
        
        for batch in dataloader:
            x = batch['X'].to(self.device)
            labels = batch['labels'].to(self.device)
            batch_index = batch.get('batch', None)
            if batch_index is not None:
                batch_index = batch_index.to(self.device)
            
            z = self.model.get_latent_representation(x, labels, batch_index, give_mean)
            all_z.append(z.cpu().numpy())
        
        return np.vstack(all_z)