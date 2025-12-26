"""对比学习损失（Node-level InfoNCE）"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeContrastiveLoss(nn.Module):
    """
    节点级对比学习损失
    
    正样本对:  同一节点的anchor view和learner view
    负样本:  当前节点与其他节点的cross-view对
    """
    
    def __init__(self, temperature: float = 0.1, negative_mode: str = 'all'):
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode  # 'all' or 'hard'
    
    def forward(self, z_anchor, z_learner):
        """
        Args:
            z_anchor: [N, D] anchor view embeddings
            z_learner:  [N, D] learner view embeddings
        
        Returns: 
            loss: scalar
        """
        N = z_anchor.size(0)
        device = z_anchor.device
        
        # L2归一化
        z_anchor = F.normalize(z_anchor, p=2, dim=-1)
        z_learner = F.normalize(z_learner, p=2, dim=-1)
        
        # 计算相似度矩阵 [N, N]
        # sim[i, j] = similarity(anchor_i, learner_j)
        sim_matrix = torch.mm(z_anchor, z_learner.t()) / self.temperature
        
        # 对角线是正样本对
        pos_sim = torch.diag(sim_matrix)  # [N,]
        
        # InfoNCE loss
        # log(exp(pos) / sum(exp(all)))
        exp_sim = torch.exp(sim_matrix)  # [N, N]
        
        if self.negative_mode == 'all':
            # 所有其他节点都是负样本
            neg_sum = exp_sim.sum(dim=1) - torch.exp(pos_sim)  # 排除自身
        elif self.negative_mode == 'hard':
            # 只用最相似的top-K作为hard negatives
            k = min(512, N - 1)
            topk_sim, _ = torch.topk(sim_matrix, k=k + 1, dim=1)  # [N, k+1]
            neg_sum = torch.exp(topk_sim[: , 1:]).sum(dim=1)  # 排除第一个（自己）
        
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sum + 1e-8))
        
        return loss.mean()