import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .embedding import PatchEmbed
from  .swin_block import SwinTransformerBlock 
from  .skipconnection.crossvit import CrossTransformer
from  .skipconnection.xcit import XCA
from  .skipconnection.xcablock import XCABlock
from  .skipconnection.xcablockdeform import XCABlockDeform
import math
import copy

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        operation for addatten
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,operation='+'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.operation =operation
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,operation=self.operation)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x,y):
       
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x,_ = blk(x,["decoder",y])
        if self.upsample is not None:
            x = self.upsample(x,None)
        
    
        return x

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x,y):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class Decoder(nn.Module):
    def __init__(self,embed_dim,depths,num_heads,window_size,\
      mlp_ratio,img_size,qkv_bias,qk_scale,drop_rate,attn_drop_rate,norm_layer,\
      use_checkpoint,num_layers,patch_size=4, in_chans=3,\
      drop_path_rate=0.1,patch_embed = None,patch_norm=True,final_upsample="expand_first",num_classes=9,args = None):
        super().__init__()
        self.patch_norm = patch_norm
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.num_classes = num_classes
        self.mode = args.mode
        self.skip_num = args.skip_num
        self.operation = args.operationaddatten
        self.add_attention = args.attention
        self.isxvit = args.iscrossvit
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.embed_dim = embed_dim
        num_patches = patch_embed.num_patches
        pretrained_dict = torch.load('./pretrained_ckpt/xcit_tiny_12_p16_224.pth', map_location='cuda')['model']
        full_dict = copy.deepcopy(pretrained_dict)

        #build cross cit 
     
        if self.mode in["swinxcit"]:
          self.Crosscit_layer = nn.ModuleList()
          for i in range(self.num_layers-1):
            
            
            XCA_layer=XCA(dim = embed_dim*2**i, num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=0,proj_drop=drop_rate,drop_path =0.1,eta=1e-5)
            model_dict = XCA_layer.state_dict()
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print(" del shape pretrain:{};shape model:{}".format(k,model_dict[k].shape))
                        del full_dict[k]
 

            XCA_layer.load_state_dict(full_dict, strict=False)
            self.Crosscit_layer.append(XCA_layer)
        #build cca block
        if self.mode in["swinxcitlpi"]:
          self.swinxcitlpi_layer = nn.ModuleList()
          for i in range(self.num_layers-1):
            XCABlock_layer = XCABlock(dim = embed_dim*2**i, num_heads=8,mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale,drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm,num_tokens=num_patches//4**i, eta=1e-5 )
            XCABlock_layer.load_state_dict(pretrained_dict, strict=False)
            self.swinxcitlpi_layer.append(XCABlock_layer)
        #deform  cca block
        if self.mode in["swinxcitlipdeform"]:
            
          self.swinxcitlipdeform_layer = nn.ModuleList()
          for i in range(self.num_layers-1):
            XCABlock_layer = XCABlockDeform(dim = embed_dim*2**i, num_heads=8,mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale,drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm,num_tokens=num_patches//4**i, eta=1e-5 )
            XCABlock_layer.load_state_dict(pretrained_dict, strict=False)
            self.swinxcitlipdeform_layer.append(XCABlock_layer)
      
        
        # build crossvit layer
        if self.mode in["crossvittoken"] or (self.mode in ['swinxcit','swinxcitlpi','swinxcitlipdeform'] and self.isxvit=='1'):
          self.Crossvit_layer = nn.ModuleList()
          for i in range(self.num_layers-1):
            self.Crossvit_layer.append(CrossTransformer(sm_dim =embed_dim*2**i, lg_dim = embed_dim*2**i, depth = 2, heads = 8, dim_head =64 , dropout = drop_rate))
      
        patches_resolution = patch_embed.patches_resolution
     
        self.patches_resolution = patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,operation =self.operation)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

       
        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)
        
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)
       
      
    def up_x4(self, x):
      H, W = self.patches_resolution
      B, L, C = x.shape
      assert L == H*W, "input features has wrong size"
      if self.final_upsample=="expand_first":
          x = self.up(x)
          x = x.view(B,4*H,4*W,-1)
          x = x.permute(0,3,1,2) #B,C,H,W
          x = self.output(x)
      return x   
           
    def forward(self, x, x_downsample,x_attention_encoder):
      for inx, layer_up in enumerate(self.layers_up):
          if inx == 0:
              if self.mode in ["swin","crossvittoken","swinxcit","swinxcitlpi","swinxcitlipdeform"]:
                  if self.add_attention=="1":
                      x = layer_up(x,x_attention_encoder[3-inx])
                  else:
                      
                      x = layer_up(x,None)   
          else:
    
              if self.mode=="swin":
                  x = torch.cat([x,x_downsample[3-inx]],-1)
                  x = self.concat_back_dim[inx](x)
                  if self.add_attention=="1":
                      if self.skip_num == '1' and inx==1:
                          x = torch.cat([x,x_downsample[3-inx]],-1)
                          x = self.concat_back_dim[inx](x)
                          x = layer_up(x,x_attention_encoder[3-inx])
                  
                      elif self.skip_num =='2' and inx in [1,2]:
                          x= torch.cat([x,x_downsample[3-inx]],-1)
                          x = self.concat_back_dim[inx](x)
                          x = layer_up(x,x_attention_encoder[3-inx])
                      
                      elif self.skip_num =='3':
                        x = torch.cat([x,x_downsample[3-inx]],-1)
                        x = self.concat_back_dim[inx](x)
                        x = layer_up(x,x_attention_encoder[3-inx])
                      else:
                        x = torch.cat([x,x_downsample[3-inx]],-1)
                        x = self.concat_back_dim[inx](x)
                        x = layer_up(x,None)
            
                  else:
                      x = layer_up(x,None)
              if self.mode =="crossvittoken":
                  if self.skip_num == '1' and inx==1:
                      y = self.Crossvit_layer[3-inx]
                      a,b= y(x,x_downsample[3-inx])
                      x = torch.cat([a,b],-1)
                      x = self.concat_back_dim[inx](x)
                  elif self.skip_num =='2' and inx in [1,2]:
                      y = self.Crossvit_layer[3-inx]
                      a,b= y(x,x_downsample[3-inx])
                      x = torch.cat([a,b],-1)
                      x = self.concat_back_dim[inx](x)
                  elif self.skip_num =='3':
                      y = self.Crossvit_layer[3-inx]
                      a,b= y(x,x_downsample[3-inx])
                      x = torch.cat([a,b],-1)
                      x = self.concat_back_dim[inx](x)
                  else:
                      x = torch.cat([x,x_downsample[3-inx]],-1)
                      x = self.concat_back_dim[inx](x)
                      #attention
                  if self.add_attention=="1":
                    x = layer_up(x,x_attention_encoder[3-inx])
                  else:
                    x = layer_up(x,None)
              
              if self.mode =="swinxcit":
                if self.skip_num == '1' and inx==1:
                  if self.isxvit=='1':
                    crcit = self.Crosscit_layer[3-inx](x_downsample[3-inx])
                    a,b = self.Crossvit_layer[3-inx](x,crcit)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                  else:
                    crcit = self.Crosscit_layer[3-inx](x_downsample[3-inx])
                    x = torch.cat([x,crcit],-1)
                    x = self.concat_back_dim[inx](x)
                elif self.skip_num =='2' and inx in [1,2]:
                    if self.isxvit=='1':
                      crcit = self.Crosscit_layer[3-inx](x_downsample[3-inx])
                      a,b = self.Crossvit_layer[3-inx](x,crcit)
                      x = torch.cat([a,b],-1)
                      x = self.concat_back_dim[inx](x)
                    else:
                      crcit = self.Crosscit_layer[3-inx](x_downsample[3-inx])
                      x = torch.cat([x,crcit],-1)
                      x = self.concat_back_dim[inx](x)

                elif self.skip_num =='3':
                  if self.isxvit=='1':
                    crcit = self.Crosscit_layer[3-inx](x_downsample[3-inx])
                    a,b = self.Crossvit_layer[3-inx](x,crcit)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                  else:
                    crcit = self.Crosscit_layer[3-inx](x_downsample[3-inx])
                    x = torch.cat([x,crcit],-1)
                    x = self.concat_back_dim[inx](x)
                    
                else: # just concat x skip  x encoder
                    x = torch.cat([x,x_downsample[3-inx]],-1)
                    x = self.concat_back_dim[inx](x)
                if self.add_attention=="1":
                    x = layer_up(x,x_attention_encoder[3-inx])
                else:
                    x = layer_up(x,None)

              if self.mode =="swinxcitlpi":
                if self.skip_num == '1' and inx==1:
                  _,token,_ = x.shape
                  token_num_sqrt = int(math.sqrt(token))
                  if self.isxvit=='1':
                    crcitlpi = self.swinxcitlpi_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    a,b = self.Crossvit_layer[3-inx](x,crcitlpi)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                    
                  else:
                    crcitlpi = self.swinxcitlpi_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    x = torch.cat([x,crcitlpi],-1)
                    x = self.concat_back_dim[inx](x)
                elif self.skip_num =='2' and inx in [1,2]:
                  _,token,_ = x.shape
                  token_num_sqrt = int(math.sqrt(token))
                  if self.isxvit=='1':
                    crcitlpi = self.swinxcitlpi_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    a,b = self.Crossvit_layer[3-inx](x,crcitlpi)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                  else:
                    crcitlpi = self.swinxcitlpi_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    x = torch.cat([x,crcitlpi],-1)
                    x = self.concat_back_dim[inx](x)
                        
                elif self.skip_num =='3':
                  _,token,_ = x.shape
                  token_num_sqrt = int(math.sqrt(token))
                  if self.isxvit=='1':
                    crcitlpi = self.swinxcitlpi_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    a,b = self.Crossvit_layer[3-inx](x,crcitlpi)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                  else:
                    crcitlpi = self.swinxcitlpi_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    x = torch.cat([x,crcitlpi],-1)
                    x = self.concat_back_dim[inx](x)
                else: 
                  x = torch.cat([x,x_downsample[3-inx]],-1)
                  x = self.concat_back_dim[inx](x)
                if self.add_attention=="1":
                      x = layer_up(x,x_attention_encoder[3-inx])
                else:
                  x = layer_up(x,None)
              if self.mode =="swinxcitlipdeform":
                if self.skip_num == '1' and inx==1:
                  _,token,_ = x.shape
                  token_num_sqrt = int(math.sqrt(token))
                  if self.isxvit=='1':
                    crcitlpi = self.swinxcitlipdeform_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    a,b = self.Crossvit_layer[3-inx](x,crcitlpi)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                    
                  else:
                    crcitlpi = self.swinxcitlipdeform_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    x = torch.cat([x,crcitlpi],-1)
                    x = self.concat_back_dim[inx](x)
                elif self.skip_num =='2' and inx in [1,2]:
                  _,token,_ = x.shape
                  token_num_sqrt = int(math.sqrt(token))
                  if self.isxvit=='1':
                    crcitlpi = self.swinxcitlipdeform_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    a,b = self.Crossvit_layer[3-inx](x,crcitlpi)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                  else:
                    crcitlpi = self.swinxcitlipdeform_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    x = torch.cat([x,crcitlpi],-1)
                    x = self.concat_back_dim[inx](x)
                        
                elif self.skip_num =='3':
                  _,token,_ = x.shape
                  token_num_sqrt = int(math.sqrt(token))
                  if self.isxvit=='1':
                    crcitlpi = self.swinxcitlipdeform_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    a,b = self.Crossvit_layer[3-inx](x,crcitlpi)
                    x = torch.cat([a,b],-1)
                    x = self.concat_back_dim[inx](x)
                  else:
                    crcitlpi = self.swinxcitlipdeform_layer[3-inx](x_downsample[3-inx],token_num_sqrt,token_num_sqrt)
                    x = torch.cat([x,crcitlpi],-1)
                    x = self.concat_back_dim[inx](x)
                else: 
                  x = torch.cat([x,x_downsample[3-inx]],-1)
                  x = self.concat_back_dim[inx](x)
                if self.add_attention=="1":
                      x = layer_up(x,x_attention_encoder[3-inx])
                else:
                  x = layer_up(x,None)
                
               
                  
              
      
      x = self.norm_up(x)  # B L C
      x = self.up_x4(x)
      return x


