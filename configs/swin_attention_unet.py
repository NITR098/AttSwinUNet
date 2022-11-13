import ml_collections
import os
import torch.nn as nn

def get_swin_unet_attention_configs():
    
    cfg = ml_collections.ConfigDict()

    # Swin unet attention Transformer Configs
    
    cfg.image_size        = 224
    cfg.patch_size        = 4
    cfg.num_classes       = 9
    cfg.in_chans          = 3
    cfg.embed_dim         = 96
    cfg.depths            = [2, 2, 6, 2]
    cfg.num_heads         = [3, 6, 12, 24]
    cfg.window_size       = 7
    cfg.mlp_ratio         = 4
    cfg.qkv_bias          = True
    cfg.qk_scale          = None
    cfg.drop_rate         = 0.0
    cfg.drop_path_rate    = 0.1
    cfg.attn_drop_rate    = 0
    cfg.ape               = False
    cfg.patch_norm        = True
    cfg.use_checkpoint    = False
    cfg.mode              = "swin"
    cfg.skip_num          = 3
    cfg.operationaddatten = '+'
    cfg.spatial_attention = '1'
    cfg.final_upsample    = "expand_first"
    cfg.norm_layer        = nn.LayerNorm
    cfg.pretrain_ckpt = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
    return cfg