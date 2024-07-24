import os
import time
import numpy as np
from sklearn import cluster

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss
from learner.cluster_contrastive_utils import cluster_contrastive_loss
import umap.umap_ as umap


class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        self.best_acc = 0
        self.best_score = None
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        self.bestStep = None
        self.gstep = 0
        self.bestStepSwitch = False
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.args.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        return token_feat

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat(
                [feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)],
                dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1),
                                        feat3['attention_mask'].unsqueeze(1)], dim=1)

        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)],
                                       dim=1)

        return input_ids.cuda(), attention_mask.cuda()

    def train_step_virtual(self, input_ids, attention_mask):

        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()

            cluster_loss = self.cluster_loss((output + 1e-08).log(), target) / output.shape[0]
            loss += 8 * cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

            # L_clu
            cluster_probs_1 = self.model.get_cluster_prob_for_lclu(embd1)
            cluster_probs_2 = self.model.get_cluster_prob_for_lclu(embd2)
            clu_loss = cluster_contrastive_loss(cluster_probs_1, cluster_probs_2, self.args.temperature)
            loss += 4 * clu_loss
            losses["clu_loss"] = clu_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    # Original code
    def train_step_explicit(self, input_ids, attention_mask):
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()

            cluster_loss = self.cluster_loss((output + 1e-08).log(), target) / output.shape[0]
            loss += 0.5 * cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

            proj1 = self.model.cluster_projection(embd1)
            proj2 = self.model.cluster_projection(embd2)
            proj3 = self.model.cluster_projection(embd3)
            cluster_cl_losses = self.model.cluster_contrastive_loss(proj1, proj2, proj3)
            loss += cluster_cl_losses
            losses["cluster_cl_loss"] = cluster_cl_losses.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        for i in np.arange(self.args.max_iter + 1):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses = self.train_step_virtual(input_ids,
                                             attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(
                input_ids, attention_mask)

            if (self.args.print_freq > 0) and ((i % self.args.print_freq == 0) or (i == self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                self.evaluate_embedding(i)
                self.model.train()

                if self.bestStepSwitch:
                    self.bestStep = i
                    self.bestStepSwitch = False

        print('---- END Saving Best Model At the: {} Step  ----'.format(self.bestStep))
        print('---- END Saving Best Model ACC: {:.3f}   ----'.format(self.best_acc))
        print('---- END Saving Best Model Clustering scores:: {}   ----'.format(self.best_score))

        return None

    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        

        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label']
                feat = self.get_batch_token(text)
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)

        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)

        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()

        kmeans = cluster.KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
        embeddings = all_embeddings.cpu().numpy()

        # Then we can use UMAP to reduce the dimension
        # waiting for code
        if self.args.rd == 'umap':
            reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', low_memory=False, n_jobs=-1,
                                force_approximation_algorithm=True)
            umap_embeddings = reducer.fit_transform(embeddings)
            kmeans.fit(umap_embeddings)
        else:
            kmeans.fit(embeddings)

        pred_labels = torch.tensor(kmeans.labels_.astype(np.int))

        # clustering accuracy 
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()

        ressave = {"acc": acc, "acc_model": acc_model}
        ressave.update(confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        np.save(self.args.resPath + 'acc_{}.npy'.format(step), ressave)
        np.save(self.args.resPath + 'scores_{}.npy'.format(step), confusion.clusterscores())
        np.save(self.args.resPath + 'mscores_{}.npy'.format(step), confusion_model.clusterscores())
        # np.save(self.args.resPath + 'mpredlabels_{}.npy'.format(step), all_pred.cpu().numpy())
        np.save(self.args.resPath + 'predlabels_{}.npy'.format(step), pred_labels.cpu().numpy())
        np.save(self.args.resPath + 'embeddings_{}.npy'.format(step), embeddings)
        # np.save(self.args.resPath + 'labels_{}.npy'.format(step), all_labels.cpu())

        # save model
        if acc_model > self.best_acc:
            self.best_acc = acc_model
            best_model_path = os.path.join(self.args.resPath, 'best_model.pth')
            torch.save(self.model, best_model_path)
            self.best_score = confusion_model.clusterscores()
            self.bestStepSwitch = True
            print('[Model] Saving ACC: {:.3f}'.format(acc_model))

        print('[Representation] Clustering scores:', confusion.clusterscores())
        print('[Representation] ACC: {:.3f}'.format(acc))
        print('[Model] Clustering scores:', confusion_model.clusterscores())
        print('[Model] ACC: {:.3f}'.format(acc_model))
        return None
