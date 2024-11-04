
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Optional, List
from torch import Tensor
import copy

from lavis.models import load_model_and_preprocess

from models.traj_former_lora import TrajFormer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _to_predict_cls_ids(
    cls_split,
    num_base,
    num_novel,
    cls_probs,
    ):
    _,num_cls = cls_probs.shape
    scores,cls_ids = torch.max(cls_probs,dim=-1)  # (n_det,)

    # 0,[1,2,....,49,50],[51,52,...,79,80]

    if cls_split == "base":
        assert num_cls == num_base  # for object class in VidOR, num_base == 50
        cls_ids += 1  # 0 ~ 49 --> 1 ~ 50,  len == 50
        
    elif cls_split == "novel":
        assert num_cls == num_novel # 30
        cls_ids += 1 + num_base    # range: 0 ~ 29  --> 51 ~ 80

    elif cls_split == "all":
        assert num_cls == num_base + num_novel  # 80
        cls_ids += 1
        # rang: 0 ~ 79 --> 1 ~ 80
    else:
        assert False, "eval_split must be base, novel, or all"

    
    return scores,cls_ids


class OpenVocTrajCls(nn.Module):

    def __init__(
        self,configs,is_train=True):
        super().__init__()
        self.is_train = is_train
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.temp_init = configs["temperature_init"]

        #**********************************************************************#
        self.class_embeddings_bert = None
        if 'text_emb_path_bert' in configs.keys():
            self.text_emb_path_bert = configs['text_emb_path_bert']
            text_embeddings_bert = torch.load(self.text_emb_path_bert).float() 
            if text_embeddings_bert.size(0) == 36 or text_embeddings_bert.size(0) == 81:
                text_embeddings_wo_bg = text_embeddings_bert[1:,:] 
            else:
                text_embeddings_wo_bg = text_embeddings_bert
            self.class_embeddings_bert = nn.Parameter(text_embeddings_wo_bg,requires_grad=False)       
        #**********************************************************************#
        
        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)

        self.traj_former = TrajFormer(configs['traj_former_cfg'])

        if self.is_train:
            self.reset_classifier_weights("base") # TODO
        else:
            pass


    def forward(self,batch_data):
        
        batch_vit_feats, batch_labels = batch_data

        batch_vit_feats = torch.cat(batch_vit_feats, dim=0) 

        batch_labels = torch.cat(batch_labels,dim=0)           

        _, traj_features = self.traj_former(None, batch_vit_feats)

        loss_dict = self.new_loss(traj_features, batch_labels, self.classifier_weights_bert, tag='traj')

        total_loss = torch.stack(list(loss_dict.values())).sum()  
        loss_for_show = {k:v.detach() for k,v in loss_dict.items()}
        loss_for_show.update({"total":total_loss.detach()})

        return  total_loss, loss_for_show

    def new_loss(self, batch_traj_embs, batch_labels, classifier_weights, tag=''):

        logits = torch.matmul(batch_traj_embs,classifier_weights.t()).mean(1) / self.temperature


        pos_mask = (0< batch_labels) & (batch_labels <= self.num_base) 
        pos_labels = batch_labels[pos_mask] -1  
        pos_logits = logits[pos_mask,:] 
        pos_cls_loss = F.cross_entropy(pos_logits, pos_labels, reduction='none')  
        
        neg_mask  = ~pos_mask
        neg_logits = logits[neg_mask,:]  
        neg_target = torch.ones_like(neg_logits) / self.num_base
        neg_cls_loss = (-1 * F.log_softmax(neg_logits,dim=-1)*neg_target).sum(dim=-1)

        d_ = logits.device
        if pos_cls_loss.numel() == 0:  
            pos_cls_loss = torch.zeros(size=(),device=d_)
        if neg_cls_loss.numel() == 0:
            pos_cls_loss = torch.zeros(size=(),device=d_)
        pos_cls_loss = pos_cls_loss.mean() * self.loss_factor["pos_cls"]
        neg_cls_loss = neg_cls_loss.mean() * self.loss_factor["neg_cls"]
        
        loss_dict = {"pos_cls_{}".format(tag):pos_cls_loss,"neg_cls_{}".format(tag):neg_cls_loss}

        return loss_dict
    
    def reset_classifier_weights(self,split):
        if split == "base":
            classifier_weights_bert = self.class_embeddings_bert[:self.num_base,:]
        elif split == "novel": 
            classifier_weights_bert =self.class_embeddings_bert[self.num_base:,:]
        elif split == "all":                  
            classifier_weights_bert = self.class_embeddings_bert
        else:
            assert False, "split must be base, novel, or all"

        self.classifier_weights_bert = classifier_weights_bert
        self.cls_split = split

    @torch.no_grad()
    def forward_inference_bsz1(self, batch_vit_feats):

        _, trans_traj_embs = self.traj_former(None, vit_feats=batch_vit_feats)

        classifier_weights_bert = self.classifier_weights_bert

        cls_logits_bert = torch.matmul(trans_traj_embs, classifier_weights_bert.t()).mean(1) / self.temperature
        cls_probs = torch.softmax(cls_logits_bert, dim=-1)    

        cls_probs_all = cls_probs
        scores,cls_ids = _to_predict_cls_ids(
            self.cls_split,
            self.num_base,
            self.num_novel,
            cls_probs_all
        )
        
        return scores,cls_ids


        

