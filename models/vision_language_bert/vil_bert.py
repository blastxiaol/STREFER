import torch
import torch.nn as nn
from external_tools.bert import VILBertForVLTasks, BertConfig

class ViLBert(nn.Module):
    def __init__(self, spatial_dim, config_path, pretrained_file=None, num_labels=1):
        super().__init__()
        config = BertConfig.from_json_file(config_path)
        if pretrained_file:
            self.vilbert = VILBertForVLTasks.from_pretrained(
                                            pretrained_file,
                                            config=config,
                                            num_labels=num_labels,
                                            default_gpu=False,
                                        )
        else:
            self.vilbert = VILBertForVLTasks(config, num_labels, default_gpu=False)
        self.vilbert.bert.v_embeddings.image_location_embeddings = nn.Linear(spatial_dim, config.v_hidden_size)

    def forward(self, token,
                        visual_feature,
                        spatial,
                        token_type_ids=None,
                        attention_mask=None,
                        image_attention_mask=None,
                        co_attention_mask=None,
                        task_ids=None,
                        output_all_encoded_layers=False,
                        output_all_attention_masks=False):
 
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, \
        vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, \
        linguisic_logit, all_attention_mask = self.vilbert(token, visual_feature, spatial,
                                                           token_type_ids, attention_mask, image_attention_mask,
                                                           co_attention_mask, task_ids,
                                                           output_all_encoded_layers, output_all_attention_masks)
        vision_logit = vision_logit.squeeze(-1)
        if not output_all_attention_masks:
            return vision_logit
        else:
            all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c = all_attention_mask
            attn_score = all_attention_mask_c[0]['attn2']
            attn_score = attn_score[:,:,:,:].mean(1).squeeze(0)
            return vision_logit, attn_score