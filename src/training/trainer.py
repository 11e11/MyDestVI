"""
训练器 - 适配新的数据格式
"""
import torch
from tqdm import tqdm


class Trainer:
    """通用训练器"""
    
    def __init__(self, model, optimizer, device='cuda'):
        """
        Args:
            model: CondSCVI或DestVI模型
            optimizer: 优化器
            device: 训练设备
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        self.history = {
            'train_loss': [],
            'reconstruction_loss': [],
            'kl_local': [],
            'kl_weight': []
        }
    
    def train_epoch(self, dataloader, kl_weight=1.0, n_obs=None):
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            kl_weight: KL散度权重
            n_obs: 总观测数（DestVI需要）
        
        Returns:
            dict: 平均损失
        """
        self.model.train()
        
        epoch_losses = {
            'loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_local': 0.0
        }
        
        for item in tqdm(dataloader, desc='Training', leave=False):
            # 数据移到设备
            item = {k: v.to(self.device) for k, v in item.items()}
            
            # 前向传播
            if n_obs is not None:
                # DestVI
                loss_dict = self.model.forward(item, kl_weight=kl_weight, n_obs=n_obs)
            else:
                # CondSCVI
                loss_dict = self.model.forward(item, kl_weight=kl_weight)
            
            loss = loss_dict['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累积损失
            epoch_losses['loss'] += loss.item()
            epoch_losses['reconstruction_loss'] += loss_dict['reconstruction_loss'].item()
            epoch_losses['kl_local'] += loss_dict['kl_local'].item()
        
        # 计算平均
        n_batches = len(dataloader)
        return {k: v / n_batches for k, v in epoch_losses.items()}
    
    def train(
        self,
        dataloader,
        n_epochs=100,
        n_epochs_kl_warmup=50,
        log_every=50,
        n_obs=None
    ):
        """
        训练循环
        
        Args:
            dataloader: 数据加载器
            n_epochs: 总epoch数
            n_epochs_kl_warmup: KL预热epoch数
            log_every: 每多少epoch打印一次
            n_obs: 总观测数（DestVI需要）
        """
        for epoch in range(n_epochs):
            # KL权重预热
            if n_epochs_kl_warmup > 0:
                kl_weight = min(1.0, epoch / n_epochs_kl_warmup)
            else:
                kl_weight = 1.0
            
            # 训练一个epoch
            losses = self.train_epoch(dataloader, kl_weight, n_obs)
            
            # 记录历史
            self.history['train_loss'].append(losses['loss'])
            self.history['reconstruction_loss'].append(losses['reconstruction_loss'])
            self.history['kl_local'].append(losses['kl_local'])
            self.history['kl_weight'].append(kl_weight)
            
            # 打印日志
            if (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(f"  Loss: {losses['loss']:.4f}")
                print(f"  Recon: {losses['reconstruction_loss']:.4f}")
                print(f"  KL: {losses['kl_local']:.4f}")
                print(f"  KL weight: {kl_weight:.4f}")
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']