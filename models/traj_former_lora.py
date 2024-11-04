
import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.models import load_model, load_model_and_preprocess
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.blip2_models.blip2 import Blip2Base
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TrajFormer(Blip2Base):

    def __init__(self, cfg):

        super().__init__()
        self.num_query_token = 32

        model, _, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=False, device='cpu')
        self.Qformer = model.Qformer
        self.tokenizer = model.tokenizer
        self.query_tokens = model.query_tokens
        self.vision_proj = model.vision_proj
        self.text_proj = model.text_proj
        self.max_txt_len = model.max_txt_len


    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width 
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def forward(self, visual_embs, vit_feats=None):

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

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )        
        
        return query_output.last_hidden_state, image_feats
    
    def encode(self, visual_embs, vit_feats=None):

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
    