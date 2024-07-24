import torch
import torch.nn.functional as F

def cluster_contrastive_loss(c_a, c_b, temperature=0.5):
    """
    Calculate cluster contrastive loss based on the provided formulas.

    Args:
    c_a (torch.Tensor): Cluster probabilities for view A, shape [batch_size, num_clusters]
    c_b (torch.Tensor): Cluster probabilities for view B, shape [batch_size, num_clusters]
    temperature (float): Temperature parameter for scaling similarities.

    Returns:
    torch.Tensor: The cluster contrastive loss.
    """
    device = c_a.device  # 获取数据所在的设备
    sim_matrix = torch.mm(c_a, c_b.T) / temperature
    exp_sim = torch.exp(sim_matrix)

    # 创建掩码，并确保其在正确的设备上
    mask = torch.eye(c_a.shape[0], device=device).bool()

    # 应用掩码以去除正样本相似度
    exp_sim = exp_sim.masked_fill(mask, 0)

    # 计算每个样本的总相似度
    all_sim = exp_sim.sum(1, keepdim=True)

    # 计算正样本相似度，只包括对角线上的元素
    pos_sim = torch.exp(torch.diag(sim_matrix)).unsqueeze(1)

    # 计算负样本相似度，即总相似度减去正样本相似度
    neg_sim = all_sim - pos_sim

    # 计算对比损失
    loss = -torch.log(pos_sim / (neg_sim + 1e-9))  # 避免除以零
    return loss.mean()