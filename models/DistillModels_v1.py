import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from utils.utils_func import load_json,trajid2pairid,sigmoid_focal_loss,unique_with_idx_nd
import math
from transformers import BertConfig,BertTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType, peft_model, PeftModel
from models.traj_former_lora import TrajFormer
from utils.config_parser import parse_config_py
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.blip2_models.Qformer import BertLMHeadModel
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from models.FinetuneModels_v1 import FinetuneQformerModels_v4
from experiments.RelationCls_VidVRD.cfg_ import model_pred_cfg

from models.modules import TransformerEncoderLayer, TransformerDecoderLayer, PositionEmbeddingSine, _get_clones


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



class DistillModel_v13(Blip2Base):
    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    
    def __init__(
        self,
        cfg,
        is_train=True,
        train_on_gt_only=True,
        cls_split='base',
        ):
        super().__init__()
        self.num_base = 71 # configs["num_base"]
        self.num_novel = 61 # configs["num_novel"]
        #***********************Load teacher model********************#
        ckpt_path_pred = cfg["teacher_path"]
        teacher_model = FinetuneQformerModels_v4(model_pred_cfg)
        print("loading check point from {}".format(ckpt_path_pred))
        check_point = torch.load(ckpt_path_pred,map_location=torch.device("cpu"))
        if "model_state_dict" in check_point.keys():
            state_dict = check_point["model_state_dict"] 
        else:
            state_dict = check_point
        teacher_model.load_state_dict(state_dict, strict=False)
        if self.training:
            assert cls_split == 'base'
        teacher_model.reset_classifier_weights(cls_split) 
        for _, param in teacher_model.named_parameters():
            param.requires_grad = False
        teacher_model.eval()
        self.teacher_model = teacher_model

        cls_weights = teacher_model.classifier_weights.detach()
        single_cls_weights = teacher_model.single_classifier_weights.detach()
        
        self.register_buffer('classifier_weights', cls_weights)
        self.register_buffer('single_classifier_weights', single_cls_weights)

        temperature = self.teacher_model.temperature
        rel_temperature = self.teacher_model.rel_temperature
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32),requires_grad=True)
        self.rel_temperature = nn.Parameter(torch.tensor([rel_temperature], dtype=torch.float32),requires_grad=True)

        #*************************************************************#
        self.Qformer, self.query_tokens = self.init_video_Qformer(
                                                num_query_token=32,  
                                                vision_width=1408, 
                                                num_hidden_layers=4
                                                )
        self.relpos2embd = self.teacher_model.relpos2embd
        self.relpos2embd_1 = self.teacher_model.relpos2embd_1
        self.relgiou2embd = self.teacher_model.relgiou2embd
        self.union2embd = self.teacher_model.union2embd
        self.union2embd_1 = nn.Sequential(
                nn.Linear(256,512),
                nn.ReLU(),
                nn.Linear(512,256)
            )        

        self.dim_hid = 1024
        self.vision_proj = nn.Linear(768,256)

        self.trajpair_proj = nn.Sequential(
            nn.Linear(512,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,512,bias=False),
        )

        self.encode_proj = nn.Sequential(
            nn.Linear(512,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,512,bias=False),
        )

        self.decode_proj = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256,bias=False),
        )

        encoder = TransformerEncoderLayer(
            256, 8, 256,
            dropout=0.1, activation='relu', normalize_before=False
        )
        self.encoder_layers = _get_clones(encoder, 3)

        decoder = TransformerDecoderLayer(
            256, 8, 256,
            dropout=0.1, activation='relu', normalize_before=False            
        )
        self.decoder_layers = _get_clones(decoder, 3)

        self.query_embed = nn.Embedding(1, 256)

        self.pos_embed = nn.Embedding(98, 256)
        self.giou_th = -0.3
        self.n_groups = 6
        self.query_size = 32
        #*************************************************************#
    
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
    
    def reset_classifier_weights(self, cls_split):

        self.teacher_model.reset_classifier_weights(cls_split)
        cls_weights = self.teacher_model.classifier_weights.detach()
        single_cls_weights = self.teacher_model.single_classifier_weights.detach()
        self.register_buffer('classifier_weights', cls_weights)
        self.register_buffer('single_classifier_weights', single_cls_weights)

    def proj_then_cls(self,s_feats,o_feats,relpos_feat,cls_split):
        
        so_feats = torch.cat([s_feats,o_feats],dim=-1)  # (N_pair,512)
        so_embds = self.trajpair_proj(so_feats)       # (N_pair,512)
        so_embds = F.normalize(so_embds,dim=-1)       # checked
        
        relpos_embds = self.relpos2embd(relpos_feat)  # (n_pair,512)
        # relpos_embds = self.teacher_model.relpos2embd(relpos_feat)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1).unsqueeze(1)

        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

        # classifier_weights = self.classifier_weights(cls_split)
        logits = torch.matmul(combined_embds,self.classifier_weights.t()).mean(1) / self.temperature  # (n_pair,num_base)

        return logits, so_embds, relpos_embds
    
    def encode(self, vit_feats):
        image_embeds = vit_feats[:, None, :]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
        return query_output.last_hidden_state, image_feats
    
    def paired_trajs(self,batch_vit_feats):
        bsz = len(batch_vit_feats)
        batch_subj_feats = []
        batch_obj_feats = []
        for bid in range(bsz):
            traj_feats = batch_vit_feats[bid]
            n_det = traj_feats.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_feats.device)   # keep the same pair_id order as that in labels
            s_feats = traj_feats[pair_ids[:,0],:]  
            o_feats = traj_feats[pair_ids[:,1],:] 
            batch_subj_feats.append(s_feats)
            batch_obj_feats.append(o_feats)
        
        batch_subj_feats = torch.cat(batch_subj_feats,dim=0)  # (N_pair,1048)
        batch_obj_feats = torch.cat(batch_obj_feats,dim=0)  # (N_pair,1048)

        return batch_subj_feats, batch_obj_feats

    def rel_giou_encode(self, batch_rel_giou):

        s_tags = batch_rel_giou[:,0] # >= giou_th  # (n_pair,)
        e_tags = batch_rel_giou[:,1] # >= giou_th
        diff_tags = (batch_rel_giou[:,1] - batch_rel_giou[:,0]) # >= 0  # (n_pair,)
        giou_tags = torch.stack([s_tags,e_tags,diff_tags],dim=-1)  # (n_pair,3)


        rel_giou_emb = F.normalize(self.relgiou2embd(giou_tags), dim=-1).unsqueeze(1)
        
        return rel_giou_emb
    
    def generate_rel_emb(self,rel_pos_emb,rel_giou_emb,batch_det_union_feats, n_det_list):
        # batch_rel_giou: (n_pair, 2)

        rel_union_feats = [b.mean(0).unsqueeze(0) for b in batch_det_union_feats]
        rel_union_feats = torch.cat([rel_union_feats[i].repeat(n_det_list[i],1) for i in range(len(rel_union_feats))], dim=0)

        # rel_feats = F.normalize(self.union2embd(rel_union_feats), dim=-1).unsqueeze(1)
        rel_feats, _ = self.encode(rel_union_feats)
        # _, rel_feats = self.encode(rel_union_feats)
        rel_feats = F.normalize(self.union2embd(rel_feats), dim=-1) # 32dim
        rel_feats = F.normalize(self.union2embd_1(rel_feats), dim=-1)

        rel_giou_emb = self.rel_giou_encode(rel_giou_emb)

        # output = torch.cat([rel_pos_emb, rel_giou_emb, rel_feats], dim=1)
        # return output
        rel_pos_emb = F.normalize(self.relpos2embd_1(rel_pos_emb), dim=-1).unsqueeze(1)

        # rel_feats = F.normalize(rel_feats+rel_pos_emb+rel_giou_emb, dim=-1)

        # if output.size(0) > 3000:
        #     k = 2
        #     output_list = torch.chunk(output, k, 0)
        #     rel_feats = torch.cat([self.trans_layer(out) for out in output_list], dim=0)
        # else:
        #     rel_feats = self.trans_layer(output)

        return rel_feats, rel_pos_emb, rel_giou_emb
    
    def split_classifier_weights(self, cls_split):

        if cls_split == "base":
            pids_list = list(range(1,self.num_base+1))   # (1,2,...,92), len==92,   exclude __background__
        elif cls_split == "novel":
            pids_list = list(range(self.num_base+1,self.num_base+self.num_novel+1))
            # (93,94,...,132), len == 40
        elif cls_split == "all":
            pids_list = list(range(1,self.num_base+self.num_novel+1))    # len==132, i.e., 1 ~ 132
        else:
            assert False, "cls_split must be base, novel, or all"
        cls_weights = self.single_predicate_classifier_weights[pids_list, :]

        return cls_weights

    def forward(self, batch_data, cls_split=""):
        return

    def loss_on_base(self,logits,labels):

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1) 

        neg_mask = labels[:,0] > 0 

        labels = labels[:,1:self.num_base+1]  

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

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
            
        return total_loss, loss_for_show



    def forward_inference_bsz1(self, data, cls_split, pred_topk=10):
        (
            vit_feats,
            union_feats,
            relpos_feat,
        )   = data


        n_det = vit_feats.shape[0]

        relpos_feat, rel_giou = relpos_feat

        _, prompt_ids, _ = self.get_giou_tags(rel_giou, self.giou_th)

        pair_ids = trajid2pairid(n_det).to(vit_feats.device)

            
        s_vit_feats = vit_feats[pair_ids[:,0],:]
        o_vit_feats = vit_feats[pair_ids[:,1],:]

        _, s_embds = self.encode(s_vit_feats)
        _, o_embds = self.encode(o_vit_feats)
        #******************************************************************************#

        bts = pair_ids.size(0)

        rel_feats, rel_pos_emb, rel_giou_emb = self.generate_rel_emb(relpos_feat, rel_giou, [union_feats], [trajid2pairid(vit_feats.size(0)).size(0)])

        input_feats = torch.cat([rel_feats, s_embds, o_embds, rel_pos_emb, rel_giou_emb], dim=1).permute(1,0,2) # (96+2, bts, 256)

        pos_embs = self.pos_embed(torch.arange(start=0, end=input_feats.size(0), step=1, dtype=torch.long).to(input_feats.device))
        pos_embs = pos_embs[:,None,:].expand(-1,bts,-1)

        for layer in self.encoder_layers:
            input_feats = layer(input_feats, pos=pos_embs)

        query_embed = self.query_embed.weight[None,:,:].expand(self.n_groups,-1,-1)[prompt_ids,:,:].permute(1,0,2)
        tgt = torch.zeros_like(query_embed)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory=input_feats, pos=pos_embs, query_pos=query_embed)

        encoded_rel_feats = input_feats[:rel_feats.size(1),:,:].permute(1,0,2)

        encoded_s_embs = input_feats[self.query_size:self.query_size*2,:,:].permute(1,0,2)
        encoded_o_embs = input_feats[self.query_size*2:self.query_size*3,:,:].permute(1,0,2)

        so_feats = torch.cat([s_embds, o_embds],dim=-1)
        so_embds = self.trajpair_proj(so_feats)       
        so_embds = F.normalize(so_embds,dim=-1)     

        relpos_embds = F.normalize(self.relpos2embd(relpos_feat), dim=-1).unsqueeze(1) 
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)


        classifier_weights = self.classifier_weights[prompt_ids,:,:]

        logits = torch.bmm(classifier_weights, combined_embds.permute(0,2,1)).mean(2) / self.temperature
        
        encode_so_emb = F.normalize(self.encode_proj(torch.cat([encoded_s_embs, encoded_o_embs], dim=-1)), dim=-1)
        combined_embds_v1 = F.normalize(so_embds+relpos_embds+encode_so_emb,dim=-1)

        logits_v1 = torch.bmm(classifier_weights, combined_embds_v1.permute(0,2,1)).mean(2) / self.temperature

        rel_output = F.normalize(rel_feats+rel_pos_emb+rel_giou_emb, dim=-1)

        
        single_classifier_weights = self.single_classifier_weights[prompt_ids,:,:]
        rel_logits = torch.bmm(single_classifier_weights, rel_output.permute(0,2,1)).mean(2) / self.rel_temperature
        

        rel_encode = F.normalize(encoded_rel_feats+tgt.permute(1,0,2), dim=-1)
        rel_feats_v1 = F.normalize(rel_feats+rel_pos_emb+rel_giou_emb+rel_encode, dim=-1)


        rel_logits_v1 = torch.bmm(single_classifier_weights, rel_feats_v1.permute(0,2,1)).mean(2) / self.rel_temperature
        #******************************************************************************#

        pred_probs = torch.sigmoid(logits) + torch.sigmoid(rel_logits) + torch.sigmoid(rel_logits_v1) + torch.sigmoid(logits_v1)

        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        
        return scores,cls_ids,pair_ids



