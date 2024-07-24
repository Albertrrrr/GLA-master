

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel
# from transformers import AutoModel, AutoTokenizer


class ClusterFeatureNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters):
        super(ClusterFeatureNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_clusters)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, cluster_centers=None, alpha=1.0, num_clusters=8, dropout_rate=0.1):
        super(SCCLBert, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size
        self.alpha = alpha
        
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

        # L_clu
        self.cluster_feature_network_for_lclu = ClusterFeatureNetwork(self.emb_size, 2048, num_clusters)

        self.cluster_projection_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))

        self.dropout_rate = dropout_rate


    
    def forward(self, input_ids, attention_mask, task_type="virtual"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        
        elif task_type == "virtual":
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            # mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2, use_random_dropout=True)


            return mean_output_1, mean_output_2
        
        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
            return mean_output_1, mean_output_2, mean_output_3
        
        else:
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [EVALUATE, VIRTUAL, EXPLICIT]")

    def get_cluster_prob_for_lclu(self, embeddings):
        # 只在计算L_clu时使用这个网络
        return self.cluster_feature_network_for_lclu(embeddings)

    def get_mean_embeddings(self, input_ids, attention_mask, use_random_dropout=False):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = bert_output[0]

        if use_random_dropout:
            # 为每个样本生成一个随机 Dropout 掩码
            # mask = torch.bernoulli(torch.full(embeddings.shape, 1 - self.dropout_rate)).to(embeddings.device)
            # embeddings = embeddings * mask  # 应用 Dropout 掩码
            dropout = torch.nn.Dropout(p=self.dropout_rate)  # 确定正确的dropout率
            embeddings = dropout(embeddings)

        mean_output = torch.sum(embeddings * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2
    
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

    def cluster_projection(self, embd):
        return self.get_cluster_prob(embd)

    def cluster_contrastive_loss(self, proj1, proj2, proj3):
        # 计算聚类级别的对比学习损失
        proj1_norm = F.normalize(proj1, dim=1)
        proj2_norm = F.normalize(proj2, dim=1)
        proj3_norm = F.normalize(proj3, dim=1)

        similarity_matrix_12 = torch.matmul(proj1_norm, proj2_norm.T)
        similarity_matrix_13 = torch.matmul(proj1_norm, proj3_norm.T)
        similarity_matrix_23 = torch.matmul(proj2_norm, proj3_norm.T)

        positives_12 = torch.diag(similarity_matrix_12)
        positives_13 = torch.diag(similarity_matrix_13)
        positives_23 = torch.diag(similarity_matrix_23)

        negatives_12 = similarity_matrix_12[~torch.eye(similarity_matrix_12.size(0), dtype=bool)].view(
            similarity_matrix_12.size(0), -1)
        negatives_13 = similarity_matrix_13[~torch.eye(similarity_matrix_13.size(0), dtype=bool)].view(
            similarity_matrix_13.size(0), -1)
        negatives_23 = similarity_matrix_23[~torch.eye(similarity_matrix_23.size(0), dtype=bool)].view(
            similarity_matrix_23.size(0), -1)

        logits_12 = torch.cat([positives_12.unsqueeze(1), negatives_12], dim=1)
        logits_13 = torch.cat([positives_13.unsqueeze(1), negatives_13], dim=1)
        logits_23 = torch.cat([positives_23.unsqueeze(1), negatives_23], dim=1)

        labels = torch.zeros(logits_12.size(0), dtype=torch.long, device=proj1.device)

        loss_12 = F.cross_entropy(logits_12 / self.alpha, labels)
        loss_13 = F.cross_entropy(logits_13 / self.alpha, labels)
        loss_23 = F.cross_entropy(logits_23 / self.alpha, labels)

        return (loss_12 + loss_13 + loss_23) / 3