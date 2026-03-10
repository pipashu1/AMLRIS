import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
class CARIS(nn.Module):
    def __init__(self, backbone, pixel_decoder, args, num_classes=1, criterion=None):
        super(CARIS, self).__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.num_classes = num_classes

        self.criterion = criterion
        self.base_lr = args.lr
        self.aug  = TextGuidedVisualEnhancer(text_dim=768, visual_dim=1024)
    def params_to_optimize(self, scale_lang=0.1, scale_vis=0.1):
        # parameters to optimize
        names_frozen = list()
        names_no_decay = list()
        lang_backbone_names_no_decay = list()
        lang_backbone_params_no_decay = list()
        lang_backbone_params_decay = list()
        backbone_names_no_decay = list()
        backbone_params_no_decay = list()
        backbone_params_decay = list()
        params_no_decay = list()
        params_decay = list()
        for name, m in self.named_parameters():
            if m.requires_grad:
                if 'backbone' in name:
                    # Language backbone
                    if 'lang_encoder' in name:
                        if 'Norm' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        elif 'embeddings' in name:
                            lang_backbone_params_no_decay.append(m)
                            lang_backbone_names_no_decay.append(name)
                        else:
                            lang_backbone_params_decay.append(m)
                    # Visual backbone
                    elif 'vis_encoder' in name:
                        if 'norm' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        elif 'position_embeddings' in name:
                            backbone_params_no_decay.append(m)
                            backbone_names_no_decay.append(name)
                        else:
                            backbone_params_decay.append(m)
                    # Others
                    elif 'lang_prompts' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
                else:
                    if 'norm' in name or 'Norm' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    elif 'prompt' in name:
                        params_no_decay.append(m)
                        names_no_decay.append(name)
                    else:
                        params_decay.append(m)
            else:
                names_frozen.append(name)

        params_to_optimize = [
            {'params': lang_backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_lang * self.base_lr},
            {'params': lang_backbone_params_decay, 'lr': scale_lang * self.base_lr},
            {'params': backbone_params_no_decay, 'weight_decay': 0.0, 'lr': scale_vis * self.base_lr},
            {'params': backbone_params_decay, 'lr': scale_vis * self.base_lr},
            {'params': params_no_decay, 'weight_decay': 0.0, 'lr': self.base_lr},
            {'params': params_decay, 'lr': self.base_lr},
        ]
        print('scale_lang_backbone: ', scale_lang)
        print('scale_vis_backbone: ', scale_vis)
        print('LANG BACKBONE NO DECAY params: ', lang_backbone_names_no_decay)
        print('BACKBONE NO DECAY params: ', backbone_names_no_decay)
        print('NO DECAY params: ', names_no_decay)
        print('FROZEN params: ', names_frozen)
        return params_to_optimize

    def forward(self, x, text, l_mask, resize_output=True, targets=None, attention_map=None,return_probs=False, return_attn=False,is_proxy=False):
        '''
            Input:
                x       [BxCxHxW]
                text    [BxN_l]
                l_mask  [BxN_l]
        '''
        input_shape = x.shape[-2:]
        lang_len = l_mask.shape[1]
        # Multi-modal encoding
        outs = self.backbone(x, text, l_mask) #vis_outs[-1]: [B, C, H, W] l_feats: [B, N_l, 768]
        vis_outs = outs[0]
        l_feats = outs[1]
        # VL pixel decoder
        l_feats = l_feats[:,:lang_len] # [B, N_l, 768]
        
        if attention_map is None:
            attention_map = self.aug(l_feats,vis_outs[-1])
        else:
            x = mask_image_with_relevance(x, attention_map, patch_size=32, threshold=0.4)
            input_shape = x.shape[-2:]
            outs = self.backbone(x, text, l_mask) #vis_outs[-1]: [B, C, H, W] l_feats: [B, N_l, 768]
            vis_outs = outs[0]

        if return_attn:
            x, attns = self.pixel_decoder(vis_outs, l_feats, l_mask, return_attn=return_attn) # [B, 1, H, W]
        else:
            x = self.pixel_decoder(vis_outs, l_feats, l_mask) # [B, 1, H, W]


        if self.training:
            if self.criterion is not None:
                losses = self.criterion(x, targets)
                if is_proxy:
                    return losses
                else:
                    return losses,attention_map

        if resize_output:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            if return_attn:
                attns = [F.interpolate(attn, size=input_shape, mode='bilinear', align_corners=True) for attn in attns]
                attns = [attn.reshape(x.shape[0], self.pixel_decoder.num_enc_layers, -1, input_shape[0], input_shape[1]) for attn in attns] # [B, N_layer, N_l, H, W]
        if x.shape[1] == 1:
            if not return_probs:
                x = x.sigmoid()
                x = (x >= 0.5) * 1
        else:
            if not return_probs:
                x = torch.argmax(x, dim=1, keepdim=True)
        if return_attn:
            return x, attns
        return x


class TextGuidedVisualEnhancer(nn.Module):
    def __init__(self, text_dim, visual_dim):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.text_proj = nn.Linear(text_dim, visual_dim)
        self.fuse_gate = nn.Linear(visual_dim, visual_dim)  

    def forward(self, text_features, visual_features):
        """
        text_features: [B, L, D_text]
        visual_features: [B, N, D_vis]
        return: enhanced_visual_features: [B, N, D_vis]
        """
        B, C, H, W = visual_features.shape
        visual_features = visual_features.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Project text to visual dim
        projected_text = self.text_proj(text_features)  # [B, L, D_vis]
        
        # Attention from visual -> text
        attention_weights = torch.bmm(visual_features, projected_text.transpose(1, 2))  # [B, N, L]
        attention_scores = F.softmax(attention_weights, dim=-1)  
        return attention_scores
    
    
def mask_image_with_relevance(x, attention_scores, patch_size=32, threshold=0.4, mask_ratio=0.33):
    """
    Args:
        x: [B, 3, H, W] 原图
        attention_scores: [B, N_patch, N_text] 
    Returns:
        x_masked: [B, 3, H, W] 
    """
    B, C, H, W = x.shape
    device = x.device
    p = patch_size
    h, w = H // p, W // p  # patch grid size
    N_patch = h * w

    relevance = attention_scores.max(dim=-1).values  # [B, N_patch]

    relevance_map = relevance.reshape(B, 1, h, w)
    relevance_map = F.interpolate(relevance_map, size=(H, W), mode='bilinear', align_corners=False)
    relevance_map = relevance_map.squeeze(1)

    img_patches = patchify(x, patch_size=p)  # [B, N_patch, p^2*3]
    rel_patches = patchify(relevance_map.unsqueeze(1), patch_size=p).squeeze(-1)  # [B, N_patch]

    img_patches_masked = img_patches.clone()
    for i in range(B):
        low_ids = torch.nonzero(rel_patches[i] < threshold, as_tuple=False).squeeze(1)
        if len(low_ids) == 0:
            continue
        low_ids = torch.clamp(low_ids, min=0, max=rel_patches[i].shape[0] - 1)
        num_to_mask = max(1, len(low_ids) // 4)
        selected = torch.randperm(len(low_ids), device=x.device)[:num_to_mask]
        img_patches_masked[i, low_ids[selected]] = 0.0
    x_masked = unpatchify(img_patches_masked, patch_size=p)
    return x_masked




def patchify(imgs, patch_size=32):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2 * C)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    n, c, h, w = imgs.shape[0], imgs.shape[1], imgs.shape[2] // p, imgs.shape[3] // p
    x = imgs.reshape(shape=(n, c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(n, h * w, p ** 2 * c))

    return x

def unpatchify(x, patch_size=32, c=3):
    """
    x: (N, L, patch_size**2 * C)
    imgs: (N, C, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    n = x.shape[0]

    x = x.reshape(shape=(n, h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(n, c, h * p, h * p))

    return imgs
