
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import contextlib
from utils.utils_func import load_json,trajid2pairid,sigmoid_focal_loss,unique_with_idx_nd
import math
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from transformers import BertConfig,BertTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType, peft_model, PeftModel
from models.traj_former_lora import TrajFormer
from utils.config_parser import parse_config_py
from experiments.TrajCls_VidVRD.traj_lora_cfg_ import model_cfg as finetune_cfg

def _to_predict_cls_ids(
    cls_split,
    num_base,
    num_novel,
    pred_probs,
    pred_topk,
    ):
    _,num_cls = pred_probs.shape
    scores,cls_ids = torch.topk(pred_probs,pred_topk,dim=-1)  

    if cls_split == "base":
        assert num_cls == num_base
        cls_ids += 1  
        
    elif cls_split == "novel":
        assert num_cls == num_novel
        cls_ids += 1 + num_base  

    elif cls_split == "all":
        assert num_cls == num_base + num_novel 
        cls_ids += 1

    else:
        assert False, "eval_split must be base, novel, or all"
    
    return scores,cls_ids

  
class FinetuneQformerModels_v4(nn.Module):

    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        temperature = configs["temperature"]  
        finetune_path = configs["finetune_path"]

        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=True)
        self.rel_temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=True)

        n_context = configs["n_context_tokens"]
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]
        
        finetune_qformer = TrajFormer(finetune_cfg)
        finetune_qformer.load_state_dict(torch.load(finetune_path, map_location=torch.device('cpu')), strict=False)

        target_module = ['value', 'query', 'text_proj', 'vision_proj']
        for name, param in finetune_qformer.named_parameters():
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=32, 
            lora_dropout=0.05, 
            bias="none",
            target_modules=target_module
        )
        finetune_qformer = get_peft_model(finetune_qformer, lora_cfg)
        self.finetune_qformer = finetune_qformer

        self.n_groups = 6
        self.giou_th = -0.3
        self.prompt_learner(n_context, pred_cls_split_info_path)

        print("*************Finetune Model Prepared.***************")
    
    def device(self):
        return list(self.parameters())[0].device


    def setup_learnable_parameters(self):

        subj_ctx_embds = torch.empty(self.n_groups, self.n_ctx, 768)
        nn.init.normal_(subj_ctx_embds, std=0.02)
        self.subj_ctx_embds = nn.Parameter(subj_ctx_embds,requires_grad=True) 
        obj_ctx_embds = torch.empty(self.n_groups, self.n_ctx, 768)
        nn.init.normal_(obj_ctx_embds, std=0.02)
        self.obj_ctx_embds = nn.Parameter(obj_ctx_embds,requires_grad=True) 

        if self.use_pos:
            self.relpos2embd = nn.Sequential(
                nn.Linear(12,256),
                nn.ReLU(),
                nn.Linear(256,512,bias=False)
            )
            self.relpos2embd_1 = nn.Sequential(
                    nn.Linear(12,128),
                    nn.ReLU(),
                    nn.Linear(128,256)
                )
            self.relgiou2embd = nn.Sequential(
                    nn.Linear(3,128),
                    nn.ReLU(),
                    nn.Linear(128,256)
                )
            self.union2embd = nn.Sequential(
                    nn.Linear(768,512),
                    nn.ReLU(),
                    nn.Linear(512,256)
                )
        
        ctx_embds = torch.empty(self.n_groups, self.n_ctx, 768)
        nn.init.normal_(ctx_embds, std=0.02)
        self.single_ctx_embds = nn.Parameter(ctx_embds,requires_grad=True)  # to be optimized
    
    def specify_clsids_range(self,split):
        if split == "base":
            pids_list = list(range(1,self.num_base+1))   
        elif split == "novel":
            pids_list = list(range(self.num_base+1,self.num_base+self.num_novel+1))
        elif split == "all":
            pids_list = list(range(1,self.num_base+self.num_novel+1))    
        else:
            assert False, "split must be base, novel, or all"
        
        return pids_list
    
    def prompt_learner(self,n_context, cls_split_info_path, use_pos=True):

        self.max_txt_len = self.finetune_qformer.max_txt_len

        self.use_pos = use_pos
        cls_split_info = load_json(cls_split_info_path)
        self.num_base = sum([v=="base" for v in cls_split_info["cls2split"].values()])
        self.num_novel = sum([v=="novel" for v in cls_split_info["cls2split"].values()])

        cls2id_map = cls_split_info["cls2id"]
        cls_names = sorted(cls2id_map.items(),key= lambda x:x[1])
        cls_names = [x[0] for x in cls_names]  
        cls_names = [name.replace("_", " ") for name in cls_names]
        name_lens = [len(name.split(" ")) for name in cls_names]
        self.n_cls = len(cls_names)
        self.n_ctx = n_context
        assert all([len_ + self.n_ctx <= self.max_txt_len for len_ in name_lens])

        place_holder_strs = " ".join(["X"] * self.n_ctx)
        token_strs = [place_holder_strs + " " + name for name in cls_names] 

        tokenizer = self.finetune_qformer.tokenizer
        batch_enc = tokenizer.batch_encode_plus(
            token_strs, 
            max_length= self.max_txt_len,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        token_ids = batch_enc.input_ids  
        token_mask = batch_enc.attention_mask 

        with torch.no_grad():
            
            token_embds = self.finetune_qformer.Qformer.bert.embeddings.word_embeddings(token_ids)

        prefix_embds =  token_embds[:, :1, :]  
        suffix_embds =  token_embds[:, 1 + self.n_ctx :, :]  

        self.register_buffer("prefix_embds", prefix_embds[None,:,:].expand(self.n_groups,-1,-1,-1).clone())
        self.register_buffer("suffix_embds", suffix_embds[None,:,:].expand(self.n_groups,-1,-1,-1).clone())
        self.register_buffer("token_mask", token_mask[None,:,:].expand(self.n_groups,-1,-1).clone())

        single_prefix_embds =  token_embds[:, :1, :]  
        single_suffix_embds =  token_embds[:, 1 + self.n_ctx :, :]  
        single_token_mask = batch_enc.attention_mask

        self.register_buffer("single_prefix_embds", single_prefix_embds[None,:,:].expand(self.n_groups,-1,-1,-1).clone())
        self.register_buffer("single_suffix_embds", single_suffix_embds[None,:,:].expand(self.n_groups,-1,-1,-1).clone())
        self.register_buffer("single_token_mask", single_token_mask[None,:,:].expand(self.n_groups,-1,-1).clone())        

        self.setup_learnable_parameters()
    
    def get_giou_tags(self,rel_gious,giou_th):

        tag_keys = torch.as_tensor(
            [[False, False, False],
            [False, False,  True],
            [False,  True,  True],
            [ True, False, False],
            [ True,  True, False],
            [ True,  True,  True]],device=rel_gious.device
        )
        
        s_tags = rel_gious[:,0] >= giou_th  # (n_pair,)
        e_tags = rel_gious[:,1] >= giou_th
        diff_tags = (rel_gious[:,1] - rel_gious[:,0]) >= 0  # (n_pair,)

        giou_tags = torch.stack([s_tags,e_tags,diff_tags],dim=-1)  # (n_pair,3)
        giou_tags_ = torch.cat([tag_keys,giou_tags],dim=0)  # (6+n_pair,3)

        uniq_tags,inverse_ids,count = torch.unique(giou_tags_,return_counts=True,return_inverse=True,dim=0,sorted=True)
        assert len(count) == 6
        inverse_ids = inverse_ids[6:]
        count = count - 1

        return giou_tags,inverse_ids,count
    
    def prompter(self, split):
        subj_list = []
        obj_list = []
        pids_list = self.specify_clsids_range(split.lower())
        n_cls = len(pids_list)
        for idx in range(self.n_groups):

            prefix = self.prefix_embds[idx,:,:,:][pids_list,:,:] 
            suffix = self.suffix_embds[idx,:,:,:][pids_list,:,:] 
            token_mask = self.token_mask[idx,:,:][pids_list,:]

            sub_ctx = self.subj_ctx_embds[idx]
            obj_ctx = self.obj_ctx_embds[idx]

            sub_ctx = sub_ctx[None,:,:].expand(n_cls, -1, -1)
            obj_ctx = obj_ctx[None,:,:].expand(n_cls, -1, -1)

            subj_token_embds = torch.cat(
                [
                    prefix,  
                    sub_ctx, 
                    suffix,  
                ],
                dim=1,
            )

            obj_token_embds = torch.cat(
                [
                    prefix, 
                    obj_ctx,   
                    suffix, 
                ],
                dim=1,
            )
            subj_list.append(self.text_encoder(subj_token_embds, token_mask).unsqueeze(0))
            obj_list.append(self.text_encoder(obj_token_embds, token_mask).unsqueeze(0))

        subj_token_embds = torch.cat(subj_list, dim=0)
        obj_token_embds = torch.cat(obj_list, dim=0)

        return subj_token_embds, obj_token_embds

    def single_prompter(self, split):

        token_list = []
        pids_list = self.specify_clsids_range(split.lower())
        n_cls = len(pids_list)

        for idx in range(self.n_groups):
            prefix = self.single_prefix_embds[idx,:,:,:][pids_list,:,:]
            suffix = self.single_suffix_embds[idx,:,:,:][pids_list,:,:]
            token_mask = self.single_token_mask[idx,:,:][pids_list,:]

            ctx = self.single_ctx_embds[idx]

            ctx = ctx[None,:,:].expand(n_cls, -1, -1)

            token_embds = torch.cat(
                [
                    prefix, 
                    ctx, 
                    suffix,  
                ],
                dim=1,
            )
            token_list.append(self.text_encoder(token_embds, token_mask).unsqueeze(0))

        token_embds = torch.cat(token_list, dim=0)
        return token_embds
    
    def text_encoder(self,token_embds,token_mask):


        bsz,max_L,_ = token_embds.shape
        assert max_L == self.max_txt_len

        text_output = self.finetune_qformer.Qformer.bert(inputs_embeds=token_embds,
                                attention_mask=token_mask,
                                return_dict=True,
                            )
        text_embeds = text_output.last_hidden_state
        text_feat = self.finetune_qformer.text_proj(text_embeds[:,0,:])
        text_feat = F.normalize(text_feat,dim=-1)                 

        return text_feat

    def generate_rel_emb(self,relpos_feats,rel_giou,union_feats):

        s_tags = rel_giou[:,0] 
        e_tags = rel_giou[:,1] 
        diff_tags = (rel_giou[:,1] - rel_giou[:,0]) 
        giou_tags = torch.stack([s_tags,e_tags,diff_tags],dim=-1)  
        rel_giou_emb = F.normalize(self.relgiou2embd(giou_tags), dim=-1).unsqueeze(1)

        union_feats = union_feats.mean(0).unsqueeze(0).repeat(rel_giou.size(0), 1)

        rel_feats, _ = self.finetune_qformer(None, union_feats)

        rel_feats = F.normalize(self.union2embd(rel_feats), dim=-1) 
        
        rel_pos_emb = F.normalize(self.relpos2embd_1(relpos_feats), dim=-1).unsqueeze(1)

        rel_feats = F.normalize(rel_feats+rel_pos_emb+rel_giou_emb, dim=-1)

        return rel_feats
    
    def forward_on_gt_only(self,batch_data):

        return

    def forward(self,batch_data,cls_split):
        if self.train_on_gt_only:
            return self.forward_on_gt_only(batch_data)

    def loss_on_base(self,logits,labels):

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1) 

        neg_mask = labels[:,0] > 0 
        labels = labels[:,1:self.num_base+1]  

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') 

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')

        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

        total_loss = pos_loss + neg_loss
        loss_for_show = {
            "total":total_loss.detach(),
            "pos_cls":pos_loss.detach(),
            "neg_cls":neg_loss.detach(),
        }
            
        return total_loss,loss_for_show


    def reset_classifier_weights(self,cls_split):

        subj_classifier_weights, obj_classifier_weights = self.prompter(cls_split)
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)  

        self.register_buffer("classifier_weights",classifier_weights,persistent=False)

        single_classifier_weights = self.single_prompter(cls_split)

        self.register_buffer("single_classifier_weights",single_classifier_weights,persistent=False)


    def forward_inference_bsz1(self, data, cls_split, pred_topk=10, traj_filter_ids=None, repro_pred_probs=None):

        (
            det_feats,
            traj_embds,
            vit_feats,
            union_feats,
            relpos_feat,
        )   = data
        if isinstance(relpos_feat, tuple):
            relpos_feat, rel_gious= relpos_feat

        if traj_filter_ids is not None:
            vit_feats = vit_feats[traj_filter_ids]

        n_det = vit_feats.shape[0]

        pair_ids = trajid2pairid(n_det).to(vit_feats.device)
        assert relpos_feat.size(0) == rel_gious.size(0) == pair_ids.size(0)

        s_vit_feats = vit_feats[pair_ids[:,0],:]
        o_vit_feats = vit_feats[pair_ids[:,1],:]

        _, s_embds = self.finetune_qformer(det_feats, s_vit_feats)
        _, o_embds = self.finetune_qformer(det_feats, o_vit_feats)

        so_embds = torch.cat([s_embds, o_embds],dim=-1) / math.sqrt(2)  

        relpos_embds = self.relpos2embd(relpos_feat)  

        relpos_embds = F.normalize(relpos_embds,dim=-1)

        combined_embds = F.normalize(so_embds+relpos_embds.unsqueeze(1),dim=-1)

        _, prompt_ids, _ = self.get_giou_tags(rel_gious,self.giou_th)

        classifier_weigths = self.classifier_weights[prompt_ids,:,:]
        logits = torch.bmm(classifier_weigths, combined_embds.permute(0,2,1)).mean(2) / self.temperature

        rel_feats = self.generate_rel_emb(relpos_feat, rel_gious, union_feats)

        rel_classifier_weights = self.single_classifier_weights[prompt_ids,:,:]
        rel_logits = torch.bmm(rel_classifier_weights, rel_feats.permute(0,2,1)).mean(2) / self.rel_temperature

        pred_probs = torch.sigmoid(logits) + torch.sigmoid(rel_logits)
        if repro_pred_probs is not None:
            pred_probs += repro_pred_probs

        scores,cls_ids = _to_predict_cls_ids(  
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        
        return scores,cls_ids,pair_ids
