import warnings

# 原模型（2.1终）
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
# 设置日志文件
# =========================== 图数据生成 ==============================
def compute_edges(node_features, threshold=0.5, seed=None):
    """
    创建固定的图结构（9节点）

    参数:
    node_features: 节点特征矩阵，形状为 (num_nodes, feature_dim)
    threshold: 未使用（为了API一致性保留）
    seed: 未使用（为了API一致性保留）

    返回:
    edge_index: PyG格式的边索引，形状为 (2, num_edges)
    """
    # 固定的边列表（节点编号从0开始）
    # 原始连接：[1,2],[2,3],[2,4],[3,4],[3,6],[4,5],[4,6],[4,8],[5,6],[5,7],[6,7],[6,8],[7,8],[8,9]
    # 转换为从0开始：[0,1],[1,2],[1,3],[2,3],[2,5],[3,4],[3,5],[3,7],[4,5],[4,6],[5,6],[5,7],[6,7],[7,8]

    edge_list = [
        # Stage 1 (索引0) 出发的边
        [0, 1], 

        [0, 3], 
        [0, 4], 
        [0, 5], 


        # Stage 2 (索引1) 出发的边
        [1, 2],  
        [1, 3],  
        [1, 4],  
        # Stage 3 (索引2) 出发的边
        [2, 3], 
        [2, 4],  
        [2, 5],  
        [2, 8],  

        # Stage 4 (索引3) 出发的边
        [3, 4],  
        [3, 5],  
        [3, 7],  
        [3, 6],  

        # Stage 5 (索引4) 出发的边
        [4, 5], 
        [4, 6], 
        [4, 8], 
        [4, 7],
        # Stage 6 (索引5) 出发的边
        [5, 6], 
        [5, 7],  

        # Stage 7 (索引6) 出发的边
        [6, 7],  #
        [6, 8],  

        # Stage 8 (索引7) 出发的边
        [7, 8], 
    ]

    # 转换为numpy数组
    edge_index = np.array(edge_list, dtype=np.int64)

    # 转换为PyG格式 (2, num_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = edge_index.to(device)
    return edge_index


def compute_aug_edges(node_features, threshold=0.5, seed=None):
    """
    创建固定的图结构（9节点），所有边方向反转

    参数:
    node_features: 节点特征矩阵，形状为 (num_nodes, feature_dim)
    threshold: 未使用（为了API一致性保留）
    seed: 未使用（为了API一致性保留）

    返回:
    edge_index: PyG格式的边索引，形状为 (2, num_edges)
    """
    # 固定的边列表（节点编号从0开始）
    # 原始连接：[1,2],[2,3],[2,4],[3,4],[3,6],[4,5],[4,6],[4,8],[5,6],[5,7],[6,7],[6,8],[7,8],[8,9]
    # 转换为从0开始：[0,1],[1,2],[1,3],[2,3],[2,5],[3,4],[3,5],[3,7],[4,5],[4,6],[5,6],[5,7],[6,7],[7,8]

    # 原始边列表（增加连接后的版本）
    edge_list = [
        # Stage 1 (索引0) 出发的边
        [0, 1],  

        [0, 3], 
        [0, 4],  
        [0, 5],  

        # Stage 2 (索引1) 出发的边
        [1, 2],  
        [1, 3], 
        [1, 4],  

        # Stage 3 (索引2) 出发的边
        [2, 3], 
        [2, 4],  
        [2, 5], 
        [2, 8],  

        # Stage 4 (索引3) 出发的边
        [3, 4],  
        [3, 5], 
        [3, 7], 
        [3, 6], 

        # Stage 5 (索引4) 出发的边
        [4, 5], 
        [4, 6], 
        [4, 8], 
        [4, 7],
        # Stage 6 (索引5) 出发的边
        [5, 6],  
        [5, 7],  

        # Stage 7 (索引6) 出发的边
        [6, 7],  
        [6, 8],  

        # Stage 8 (索引7) 出发的边
        [7, 8],  
    ]

    # 反转所有边的方向（交换源节点和目标节点）
    reversed_edge_list = [[edge[1], edge[0]] for edge in edge_list]

    # 转换为numpy数组
    edge_index = np.array(reversed_edge_list, dtype=np.int64)

    # 转换为PyG格式 (2, num_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index = edge_index.to(device)
    return edge_index


# 新增：计算上游节点
edge_list = [
    # Stage 1 (索引0) 出发的边
    [0, 1],  

    [0, 3], 
    [0, 4],  
    [0, 5], 

    # Stage 2 (索引1) 出发的边
    [1, 2],  
    [1, 3],  
    [1, 4],  

    # Stage 3 (索引2) 出发的边
    [2, 3], 
    [2, 4],  
    [2, 5],  
    [2, 8],  

    # Stage 4 (索引3) 出发的边
    [3, 4], 
    [3, 5],  
    [3, 7],  
    [3, 6],  

    # Stage 5 (索引4) 出发的边
    [4, 5], 
    [4, 6],  
    [4, 8],  
    [4, 7],
    # Stage 6 (索引5) 出发的边
    [5, 6],  
    [5, 7],  

    # Stage 7 (索引6) 出发的边
    [6, 7],  
    [6, 8],  

    # Stage 8 (索引7) 出发的边
    [7, 8], 
]
in_neighbors = defaultdict(list)
for src, dst in edge_list:
    in_neighbors[dst].append(src)



def generate_laser_graph(num_nodes=9, num_features=6, num_label=6, seed=None):
    df = pd.read_csv('./data/', header=0, encoding='utf-8')
    df_x = df.iloc[:, :num_nodes * num_features]
    df_y = df.iloc[:, num_nodes * num_features:]
    df_x = np.array(df_x).reshape(-1, num_nodes, num_features)
    df_y = np.array(df_y).reshape(-1, num_nodes, num_label)
    df_x = torch.tensor(df_x, dtype=torch.float32)
    df_y = torch.tensor(df_y, dtype=torch.float32)
    x_list = torch.unbind(df_x, dim=0)
    y_list = torch.unbind(df_y, dim=0)
    data_list = []

    for i in range(df.shape[0]):
        x = x_list[i].to(device)
        y = y_list[i].to(device)
        noise = torch.normal(mean=0.0, std=0.05, size=x.shape).to(device)
        x_perturbed = x + noise
        edge_index = compute_edges(x_perturbed.cpu().numpy(), threshold=0.65, seed=seed if seed is not None else i)
        data = Data(x=x, edge_index=edge_index, y=y)

        # 新增：在这里添加 subgraph_masks
        data.subgraph_masks = []
        for center in range(num_nodes):
            nodes_in_subgraph = [center] + in_neighbors[center]
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            mask[torch.tensor(nodes_in_subgraph, device=device)] = True
            data.subgraph_masks.append(mask)

        data_list.append(data)
        print(f"图 {i + 1}： 边数： {edge_index.shape[1]}")

    return data_list


def generate_augmented_laser_graph(num_nodes=9, num_features=6, num_label=6, seed=None):
    df = pd.read_csv('./data/', header=0, encoding='utf-8')
    df_x = df.iloc[:, :num_nodes * num_features]
    df_y = df.iloc[:, num_nodes * num_features:]
    df_x = np.array(df_x).reshape(-1, num_nodes, num_features)
    df_y = np.array(df_y).reshape(-1, num_nodes, num_label)
    df_x = torch.tensor(df_x, dtype=torch.float32)
    df_y = torch.tensor(df_y, dtype=torch.float32)
    x_list = torch.unbind(df_x, dim=0)
    y_list = torch.unbind(df_y, dim=0)
    data_list = []

    for i in range(df.shape[0]):
        x = x_list[i].to(device)
        y = y_list[i].to(device)
        noise = torch.normal(mean=0.0, std=0.1, size=x.shape).to(device)
        x_perturbed = x + noise
        edge_index = compute_aug_edges(x_perturbed.cpu().numpy(), threshold=0.65, seed=seed if seed is not None else i)
        data = Data(x=x_perturbed, edge_index=edge_index, y=y)

        # 新增：在这里添加 subgraph_masks
        data.subgraph_masks = []
        for center in range(num_nodes):
            nodes_in_subgraph = [center] + in_neighbors[center]
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            mask[torch.tensor(nodes_in_subgraph, device=device)] = True
            data.subgraph_masks.append(mask)

        data_list.append(data)
        print(f"增强图 {i + 1}： 边数： {edge_index.shape[1]}")

    return data_list


# =========================== Transformer Encoder ==============================
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.3, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.transformer(x, mask=mask)
        return self.norm(x)


# =========================== 空间注意力 ==============================
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.query_conv = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.key_conv = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.value_conv = nn.Linear(in_channels, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        query = self.query_conv(x).unsqueeze(0)
        key = self.key_conv(x).unsqueeze(0)
        value = self.value_conv(x)

        energy = torch.bmm(query, key.transpose(1, 2))
        attention = self.softmax(energy / (x.size(-1) ** 0.5))

        row, col = edge_index
        edge_mask = torch.zeros(1, num_nodes, num_nodes, device=x.device)
        edge_mask[0, row, col] = 1.0
        attention = attention * edge_mask
        attention = self.softmax(attention)

        out = torch.bmm(attention, value.unsqueeze(0)).squeeze(0)
        out = self.gamma * out + x
        return out


# =========================== Latent Projector ==============================
class LatentProjector(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# =========================== Latent Diffusion Refiner ==============================
class LatentDiffusionRefiner(nn.Module):
    def __init__(self, latent_dim, num_steps=12, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1. - alpha_bars))

        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        self.denoise_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, z_noisy, t):
        t_norm = t.float() / (self.num_steps - 1)
        t_embed = self.time_embed(t_norm.unsqueeze(-1))
        inp = torch.cat([z_noisy, t_embed], dim=-1)
        return self.denoise_net(inp)

    @torch.no_grad()
    def refine(self, z_start, num_steps=None, eta=0.0):
        if num_steps is None:
            num_steps = self.num_steps
        device = z_start.device
        z = z_start.clone()
        N = z.shape[0]

        for i in reversed(range(num_steps)):
            t = torch.full((N,), i, device=device, dtype=torch.long)
            pred_noise = self(z, t)

            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]

            pred_z0 = (z - self.sqrt_one_minus_alpha_bars[i] * pred_noise) / torch.sqrt(alpha_bar)
            sigma = eta * torch.sqrt(
                (1 - self.alpha_bars[i - 1] if i > 0 else 0) /
                (1 - alpha_bar) * self.betas[i]
            )

            direction = torch.sqrt(1 - alpha - sigma ** 2) * pred_noise
            noise_term = sigma * torch.randn_like(z) if i > 0 else 0

            z = torch.sqrt(alpha) * pred_z0 + direction + noise_term

        return z


# =========================== GNNRegressor（加入latent diffusion） ==============================
class GNNRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_output,
                 num_transformer_layers=1, num_heads=4,
                 latent_dim=64, diffusion_steps=12, diffusion_beta_end=0.02):
        super().__init__()

        self.conv1 = ChebConv(in_channels, hidden_channels, K=2)


        self.attention1 = SpatialAttention(hidden_channels)


        self.transformer = TransformerEncoder(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 2,
            dropout=0.3,
            num_layers=num_transformer_layers
        )

        # Latent diffusion 组件
        self.latent_projector = LatentProjector(hidden_channels, latent_dim=latent_dim)
        self.latent_diffuser = LatentDiffusionRefiner(latent_dim=latent_dim,
                                                      num_steps=diffusion_steps,
                                                      beta_end=diffusion_beta_end)

        self.lin = nn.Linear(hidden_channels, num_output)
        self.latent_scale = nn.Parameter(torch.tensor(0.3))  # 初始小一点
        # 控制参数
        self.use_latent_diff = True
        self.latent_diff_prob = 0.0  # 初始0，训练后期增加
        self.latent_inference_steps = 8

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.attention1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)

        if self.use_latent_diff:
            recon, z = self.latent_projector(x)

            if self.training:
                if torch.rand(1).item() < self.latent_diff_prob:
                    z_refined = self.latent_diffuser.refine(z, num_steps=self.latent_inference_steps)
                else:
                    z_refined = z
            else:
                z_refined = self.latent_diffuser.refine(z, num_steps=self.latent_inference_steps)

            refined_x = self.latent_projector.decode(z_refined)

            # 关键改动：残差 + 可学习缩放
            scale = torch.sigmoid(self.latent_scale) if hasattr(self, 'latent_scale') else 0.3
            x = x + scale * refined_x  # 残差连接，防止完全覆盖原始特征



        x = x.unsqueeze(0)  # [1, N, D]
        x = self.transformer(x)
        x = x.squeeze(0)  # [N, D]

        return self.lin(x).squeeze(-1)

    def get_embedding(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.attention1(x, edge_index)


        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)

        if self.use_latent_diff:
            recon, z = self.latent_projector(x)
            z_refined = self.latent_diffuser.refine(z, num_steps=self.latent_inference_steps)
            x = self.latent_projector.decode(z_refined)

        return x


# =========================== Tower ==============================
class Tower(nn.Module):
    activation_classes = {'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout,
                 batch_norm=False, layer_norm=False, activation='relu'):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(num_layers + 1):
            in_dim = input_size if i == 0 else hidden_size
            out_dim = hidden_size if i < num_layers else output_size
            self.layers.add_module(f"{i}-Linear", nn.Linear(in_dim, out_dim))
            if i < num_layers:
                self.layers.add_module(f"{i}-{activation}", self.activation_classes[activation.lower()]())
                self.layers.add_module(f"{i}-Dropout", nn.Dropout(dropout))
                if batch_norm:
                    self.layers.add_module(f"{i}-BatchNorm1d", nn.BatchNorm1d(hidden_size))
                if layer_norm:
                    self.layers.add_module(f"{i}-LayerNorm", nn.LayerNorm(hidden_size))

    def forward(self, x):
        return self.layers(x)


# =========================== NT-Xent 对比损失 ==============================
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.t())

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    eye = torch.eye(2 * batch_size, device=z1.device)
    mask = (eye == 0).float()

    similarity_matrix = similarity_matrix / temperature
    exp_sim = torch.exp(similarity_matrix) * mask
    denominator = exp_sim.sum(dim=1)
    loss = -torch.log(torch.exp(positives / temperature) / denominator + 1e-8)
    return loss.mean()


# 新增：三层级对比损失（节点 + 子图 + 图）
def triple_level_contrastive_loss(z_node1, z_node2, z_sub1, z_sub2, z_graph1, z_graph2,
                                  temperature=0.5, lambda_node=1.0, lambda_sub=0.3, lambda_graph=1.0):
    # 节点级 NT-Xent
    node_loss = nt_xent_loss(z_node1, z_node2, temperature)

    # 子图级 NT-Xent
    sub_loss = nt_xent_loss(z_sub1, z_sub2, temperature)

    # 图级 NT-Xent
    graph_loss = nt_xent_loss(z_graph1, z_graph2, temperature)

    return lambda_node * node_loss + lambda_sub * sub_loss + lambda_graph * graph_loss


# =========================== PseudoLabelFusion ==============================
class PseudoLabelFusion(nn.Module):
    def __init__(self, feature_dim, label_dim):
        super().__init__()
        self.fusion = nn.Linear(feature_dim + label_dim, feature_dim)
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 可学习融合比例

    def forward(self, x, pseudo_y):
        # x: [feature_dim], pseudo_y: [label_dim]
        combined = torch.cat([x, pseudo_y], dim=-1)
        fused = self.fusion(combined)
        return self.alpha * x + (1 - self.alpha) * fused


# =========================== MMOE ==============================
class MMOE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_tasks, num_experts, tower_hidden_size,
                 dropout=0.3, use_contrastive=True, proj_dim=128, num_transformer_layers=1, num_heads=4,
                 latent_dim=128, diffusion_steps=12, diffusion_beta_end=0.01,
                 num_nodes=9):  # 新增：显式传入 num_nodes
        super().__init__()

        self.num_nodes = num_nodes  # 固定为9
        self.num_tasks = num_tasks
        self.hidden_channels = hidden_channels

        # 伪标签融合层
        self.pseudo_fusion = PseudoLabelFusion(in_channels, num_tasks)
        self.num_experts = num_experts

        # 专家：每个专家输出 hidden_channels，而不是直接输出任务值
        self.experts = nn.ModuleList([
            GNNRegressor(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_output=hidden_channels,  # ← 关键：改成输出 hidden，而不是1或6
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads,
                latent_dim=latent_dim,
                diffusion_steps=diffusion_steps,
                diffusion_beta_end=diffusion_beta_end
            )
            for _ in range(num_experts)
        ])

        # 每个节点一个 gate：输入 hidden → num_experts
        self.gates = nn.ModuleList([
            nn.Linear(hidden_channels, num_experts) for _ in range(self.num_nodes)
        ])

        # 每个节点一个 tower：输入 hidden → num_tasks (6)
        self.towers = nn.ModuleList([
            Tower(
                input_size=hidden_channels,
                hidden_size=tower_hidden_size,
                output_size=num_tasks,  # ← 输出6个任务
                num_layers=2,
                dropout=dropout,
                activation='relu'
            ) for _ in range(self.num_nodes)
        ])

        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.graph_readout = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, proj_dim)  # 通常 128 或 256
            )
            self.node_proj = nn.Sequential(  # 用于节点级对比的投影头
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, proj_dim)
            )
            self.subgraph_proj = nn.Sequential(  # 新增：用于子图级对比的投影头
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, proj_dim)
            )

    def forward(self, data):
        # data.x: [total_nodes, in_channels]   total_nodes = batch_size * num_nodes
        total_nodes = data.x.size(0)
        batch_size = total_nodes // self.num_nodes

        # ------------------ 专家前向 ------------------
        expert_outputs = []
        for expert in self.experts:
            out = expert(data)  # [total_nodes, hidden_channels]
            expert_outputs.append(out)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [total_nodes, num_experts, hidden]

        # gate 输入：可以选择第一个专家、平均、或 data.x 本身，这里用第一个专家作为示例
        gate_input = expert_outputs[:, 0, :]  # [total_nodes, hidden]

        # ------------------ 逐节点位置计算 gate weights ------------------
        gate_weights_list = []
        for node_idx in range(self.num_nodes):
            # 取出这个节点在所有 batch 中的特征（stride = num_nodes）
            node_gate_input = gate_input[node_idx::self.num_nodes]  # [batch_size, hidden]
            gate_logits = self.gates[node_idx](node_gate_input)  # [batch_size, num_experts]
            gate_w = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]
            gate_weights_list.append(gate_w)

        gate_weights = torch.stack(gate_weights_list, dim=1)  # [batch_size, num_nodes, num_experts]

        # ------------------ 加权融合专家输出 ------------------
        # expert_outputs: [total_nodes, num_experts, hidden]
        # reshape 为 [batch_size, num_nodes, num_experts, hidden]
        expert_reshaped = expert_outputs.view(batch_size, self.num_nodes, self.num_experts, -1)

        # einsum 加权求和 → [batch_size, num_nodes, hidden]
        fused = torch.einsum('bne,bnef->bnf', gate_weights, expert_reshaped)

        # ------------------ 逐节点 tower 前向 ------------------
        outputs_list = []
        for node_idx in range(self.num_nodes):
            node_fused = fused[:, node_idx, :]  # [batch_size, hidden]
            node_pred = self.towers[node_idx](node_fused)  # [batch_size, num_tasks]
            outputs_list.append(node_pred)

        # stack 成 [batch_size, num_nodes, num_tasks]
        outputs = torch.stack(outputs_list, dim=1)  # [batch_size, 9, 6]

        # 展平回 [total_nodes, num_tasks] 与 mask/y 匹配
        return outputs.view(total_nodes, self.num_tasks)

    def get_node_embeddings(self, data):
        # 返回 [total_nodes, hidden_channels]   ← 原始 GNN 输出的节点表征
        with torch.no_grad():
            emb = self.experts[0].get_embedding(data)  # 或融合所有专家后的 fused
            return emb

    def get_subgraph_embeddings(self, data):
        # 返回 [num_nodes, hidden_channels] 或 [num_nodes, proj_dim]
        node_emb = self.get_node_embeddings(data)  # [total_nodes, hidden]
        subgraph_embs = []
        for mask in data.subgraph_masks:  # 9个 mask
            sub_emb = node_emb[mask]  # [sub_size, hidden]
            sub_emb = sub_emb.mean(dim=0)  # mean pool
            subgraph_embs.append(sub_emb)
        return torch.stack(subgraph_embs)  # [9, hidden]

    def get_graph_embedding(self, data):
        # 返回 [batch_size, proj_dim]
        node_emb = self.get_node_embeddings(data)  # [total_nodes, hidden]
        node_emb = self.node_proj(node_emb)  # 可选：节点先投影
        batch = data.batch  # 必须有 batch 信息！
        graph_emb = global_mean_pool(node_emb, batch)  # 或 global_add_pool / attention
        graph_emb = self.graph_readout(graph_emb)  # [batch_size, proj_dim]
        return graph_emb
# =========================== 新增：融合函数 ==========================
def fuse_features_with_upstream_labels(data, upstream_labels_dict, fusion_layer, num_nodes=9):
    """
    根据上游标签（可以是真实 y 或模型预测的伪标签）对每个节点的特征进行融合
    upstream_labels_dict: dict {node_idx: tensor[6]}
    """
    # 克隆一份，避免 inplace 破坏梯度
    new_x = data.x.clone()

    for node_i in range(num_nodes):
        upstream_nodes = in_neighbors[node_i]
        if not upstream_nodes:
            continue

        # 收集上游标签
        upstream_vals = [upstream_labels_dict[up] for up in upstream_nodes if up in upstream_labels_dict]
        if not upstream_vals:
            continue

        # 简单平均
        agg_label = torch.mean(torch.stack(upstream_vals), dim=0)

        # 当前节点特征
        x_i = data.x[node_i]

        # 融合
        fused_x = fusion_layer(x_i, agg_label)

        # 修改副本
        new_x[node_i] = fused_x

    # 替换原 tensor
    data.x = new_x


# =========================== 训练主函数 ==============================
def train_and_evaluate():
    # 确保模型在正确的设备上
    print(f"训练使用的设备: {device}")
    num_nodes = 9
    # 在这块进行更改data.x,并进行特征融合降维
    # 超参数
    in_channels = dataset[0].num_features
    hidden_channels = 128
    num_tasks = 6
    num_experts = 2
    tower_hidden_size = 128
    batch_size = 12
    epochs = 1
    lr = 0.001
    weight_decay = 1e-5
    patience = 40
    num_transformer_layers = 1
    num_heads = 4
    latent_dim = 128

    train_losses = []
    val_losses = []
    # 增强数据
    augmented_dataset = generate_augmented_laser_graph()

    # 模型
    model = MMOE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_tasks=num_tasks,
        num_experts=num_experts,
        tower_hidden_size=tower_hidden_size,
        num_nodes=9,
        dropout=0.3,
        use_contrastive=True,
        proj_dim=128,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        latent_dim=latent_dim,
        diffusion_steps=12,
        diffusion_beta_end=0.01
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)

    # 生成完整数据集（原始 + 增强，行序完全一致）
    dataset = generate_laser_graph()
    augmented_dataset = generate_augmented_laser_graph()
    task_weights = load_task_weights('./data/task_weights.csv', num_tasks=6)
    print(f"任务权重: {task_weights}")
    # 从外部文件读取划分索引
    train_idx, test_idx = load_split_indices('./data/split_indices.csv')

    # 用同一套索引切分原始集和增强集，保证视图配对
    train_data, test_data = split_by_indices(dataset, train_idx, test_idx)
    train_aug_data, _ = split_by_indices(augmented_dataset, train_idx, test_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_aug_loader = DataLoader(train_aug_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    alpha = 0.3  # 对比损失权重
    recon_weight = 0.1  # 重建损失权重

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_aug_iter = iter(train_aug_loader)

        # 逐渐开启 latent diffusion
        if epoch >= 80:
            for expert in model.experts:
                expert.latent_diff_prob = min(0.85, 0.05 + (epoch - 80) * 0.008)

        # 对比学习权重启动策略（早期偏局部，后期偏全局）
        if epoch < 100:
            lambda_node = 1.0
            lambda_sub = 0.3
            lambda_graph = 0.3
        elif epoch < 300:
            lambda_node = 0.8
            lambda_sub = 0.8
            lambda_graph = 0.8
        else:
            lambda_node = 0.6
            lambda_sub = 1.0
            lambda_graph = 1.2

        for data in train_loader:
            data = data.to(device)
            try:
                data_aug = next(train_aug_iter)
                data_aug = data_aug.to(device)
            except StopIteration:
                train_aug_iter = iter(train_aug_loader)
                data_aug = next(train_aug_iter)
                data_aug = data_aug.to(device)

            optimizer.zero_grad()

            # 训练时：用真实标签做上游融合
            true_labels_dict = {i: data.y[i] for i in range(num_nodes)}
            fuse_features_with_upstream_labels(data, true_labels_dict, model.pseudo_fusion, num_nodes=num_nodes)

            outputs = model(data)

            loss = 0
            for task_idx in range(num_tasks):
                task_loss = criterion(outputs[data.train_mask, task_idx],
                                      data.y[data.train_mask, task_idx])
                loss += task_weights[task_idx] * task_loss

            # 添加 latent projector 的重建损失（端到端）
            x_trans = model.experts[0].get_embedding(data)
            recon, _ = model.experts[0].latent_projector(x_trans)
            recon_loss = criterion(recon, x_trans)
            loss += recon_weight * recon_loss

            if model.use_contrastive:
                # ─────── 节点级对比 ───────
                node_emb1 = model.get_node_embeddings(data)
                node_emb2 = model.get_node_embeddings(data_aug)
                node_proj1 = model.node_proj(node_emb1)
                node_proj2 = model.node_proj(node_emb2)

                # ─────── 子图级对比 ───────
                sub_emb1 = model.get_subgraph_embeddings(data)
                sub_emb2 = model.get_subgraph_embeddings(data_aug)
                sub_proj1 = model.subgraph_proj(sub_emb1)
                sub_proj2 = model.subgraph_proj(sub_emb2)

                # ─────── 图级对比 ───────
                graph_emb1 = model.get_graph_embedding(data)
                graph_emb2 = model.get_graph_embedding(data_aug)

                # 三层融合损失
                contrast_loss = triple_level_contrastive_loss(
                    node_proj1, node_proj2,
                    sub_proj1, sub_proj2,
                    graph_emb1, graph_emb2,
                    temperature=0.5,
                    lambda_node=lambda_node,
                    lambda_sub=lambda_sub,
                    lambda_graph=lambda_graph
                )

                loss += alpha * contrast_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1:3d}/{epochs} train loss: {avg_train_loss:.6f} "
              f"latent diff prob: {model.experts[0].latent_diff_prob:.3f}")

        # 验证
        model.eval()
        total_val_loss = 0
        test_preds = [[] for _ in range(num_nodes)]
        test_trues = [[] for _ in range(num_nodes)]
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                # 先预测一次得到伪标签
                pseudo_outputs = model(data)

                # 用伪标签做融合
                pseudo_labels_dict = {i: pseudo_outputs[i] for i in range(num_nodes)}
                fuse_features_with_upstream_labels(data, pseudo_labels_dict, model.pseudo_fusion, num_nodes=num_nodes)

                # 再预测一次得到最终输出
                final_outputs = model(data)

                val_loss = 0
                for task_idx in range(num_tasks):
                    task_loss = criterion(final_outputs[data.test_mask, task_idx], data.y[data.test_mask, task_idx])
                    val_loss += task_weights[task_idx] * task_loss
                total_val_loss += val_loss.item()

                for node_idx in range(num_nodes):
                    if data.test_mask[node_idx]:
                        test_preds[node_idx].append(final_outputs[node_idx].detach().cpu().numpy())
                        test_trues[node_idx].append(data.y[node_idx].detach().cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        print(f"第 {epoch + 1} 轮验证损失： {avg_val_loss:.5f}")

        # 输出训练过程中测试阶段每个节点每个任务的指标
        print("【训练过程中测试阶段各节点各任务指标】")
        for node_idx in range(num_nodes):
            print(f"节点 {node_idx + 1}:")
            for task_idx in range(num_tasks):
                if test_preds[node_idx]:
                    pred_arr = np.array(test_preds[node_idx])
                    true_arr = np.array(test_trues[node_idx])
                    try:
                        r2 = r2_score(true_arr[:, task_idx], pred_arr[:, task_idx])
                        mse = mean_squared_error(true_arr[:, task_idx], pred_arr[:, task_idx])
                        mae = mean_absolute_error(true_arr[:, task_idx], pred_arr[:, task_idx])
                        pearson, _ = pearsonr(true_arr[:, task_idx], pred_arr[:, task_idx]) if len(true_arr) > 1 else (
                        0.0, 0.0)
                    except Exception:
                        r2, mse, mae, pearson = 0.0, 0.0, 0.0, 0.0

                    print(f" 任务 {task_idx + 1}: R² {r2:.4f}, MSE {mse:.4f}, MAE {mae:.4f}, Pearson {pearson:.4f}")
                else:
                    print(f" 任务 {task_idx + 1}: 无测试数据")
            print("=" * 30)

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"在第 {epoch + 1} 轮早停")
            model.load_state_dict(best_model_state)
            break

    # ===== 最终测试阶段 =====
    model.eval()
    node_preds = [[[] for _ in range(num_tasks)] for _ in range(num_nodes)]
    node_trues = [[[] for _ in range(num_tasks)] for _ in range(num_nodes)]
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # 先预测一次得到伪标签
            pseudo_outputs = model(data)

            # 用伪标签做融合
            pseudo_labels_dict = {i: pseudo_outputs[i] for i in range(num_nodes)}
            fuse_features_with_upstream_labels(data, pseudo_labels_dict, model.pseudo_fusion, num_nodes=num_nodes)

            # 再预测一次得到最终输出
            final_outputs = model(data)

            for node_idx in range(num_nodes):
                if data.test_mask[node_idx]:
                    for task_idx in range(num_tasks):
                        pred = final_outputs[node_idx, task_idx].cpu().numpy()
                        true = data.y[node_idx, task_idx].cpu().numpy()
                        node_preds[node_idx][task_idx].append(pred)
                        node_trues[node_idx][task_idx].append(true)

    print("【最终测试阶段各节点各任务指标】")
    for node_idx in range(num_nodes):
        print(f"\n节点 {node_idx + 1}:")
        for task_idx in range(num_tasks):
            if node_preds[node_idx][task_idx]:
                pred_arr = np.array([pred for pred in node_preds[node_idx][task_idx]])
                true_arr = np.array([true for true in node_trues[node_idx][task_idx]])
                try:
                    r2 = r2_score(true_arr, pred_arr)
                    mse = mean_squared_error(true_arr, pred_arr)
                    mae = mean_absolute_error(true_arr, pred_arr)
                    pearson, _ = pearsonr(true_arr, pred_arr) if len(true_arr) > 1 else (0.0, 0.0)
                except Exception:
                    r2, mse, mae, pearson = 0.0, 0.0, 0.0, 0.0
                print(f" 任务 {task_idx + 1}: R² {r2:.4f}, MSE {mse:.4f}, MAE {mae:.4f}, Pearson {pearson:.4f}")
            else:
                print(f" 任务 {task_idx + 1}: 无测试数据")

    # 最终评估（这里省略详细指标计算代码，与原版类似）
    print("\n训练完成，最佳验证损失:", best_val_loss)

