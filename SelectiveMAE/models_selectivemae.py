# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from functools import partial
import math
import torch
from transformer_utils import Block, CrossAttentionBlock, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
import torch.nn as nn
from token_selected_smooth import TokenSelect_smooth as TokenSelect


class HOGLayerC(nn.Module):
    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16,
                 norm_out: bool = False,
                 in_channels: int = 3) -> None:
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        self.in_channels = in_channels
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(self.in_channels, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = self.get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer('gkern', gkern)
        self.norm_out = norm_out

    def get_gkern(self, kernlen: int, std: int) -> torch.Tensor:

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n ** 2)
            return w

        gkern1d = _gaussian_fn(kernlen, std)
        gkern2d = gkern1d[:, None] * gkern1d[None, :]
        return gkern2d / gkern2d.sum()

    def _reshape(self, hog_feat: torch.Tensor) -> torch.Tensor:
        hog_feat = hog_feat.flatten(1, 2)
        unfold_size = hog_feat.shape[-1] // 14
        hog_feat = (
            hog_feat.permute(0, 2, 3,
                             1).unfold(1, unfold_size, unfold_size).unfold(
                2, unfold_size,
                unfold_size).flatten(1, 2).flatten(2))
        return hog_feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=self.in_channels)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=self.in_channels)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins
        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        if self.norm_out:
            out = F.normalize(out, p=2, dim=2)
        out_1d = self._reshape(out)
        mean = out_1d.mean(dim=-1, keepdim=True)
        var = out_1d.var(dim=-1, keepdim=True)
        out_1d = (out_1d - mean) / (var + 1.e-6) ** .5
        out = out_1d.mean(dim=-1)
        return out



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 weight_fm=False,
                 use_fm=[-1], use_input=False, self_attn=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.token_select = TokenSelect()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.use_input = use_input
        if len(use_fm) == 1 and use_fm[0] == -1:
            self.use_fm = list(range(depth))
        else:
            self.use_fm = [i if i >= 0 else depth + i for i in use_fm]
        # --------------------------------------------------------------------------
        self.decoder_embed1 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                                qk_scale=None, norm_layer=norm_layer, self_attn=self_attn)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        # encoder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),requires_grad=False)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),requires_grad=False)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        # Probability prediction network
        self.num_patches = num_patches
        self.pos_embed_probs = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.get_token_probs = nn.Sequential(
            Block(dim=embed_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                  drop=0.1, attn_drop=0.00, drop_path=0.00, norm_layer=nn.LayerNorm),
            nn.Linear(embed_dim, 1),
            torch.nn.Flatten(start_dim=1),
        )
        self.get_token_probs_linear = nn.Linear(embed_dim, 1)
        self.get_token_probs_flatten = torch.nn.Flatten(start_dim=1)
        self.hog = HOGLayerC(nbins=9, pool=8, norm_out=True, in_channels=3)
        self.softmax = nn.Softmax(dim=-1)


    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, kept_mask_ratio, imgs):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        len_masked_reconstruct = int(L * (kept_mask_ratio))
        len_selected_initialization = int(len_keep / 2)
        with torch.no_grad():
            logits = self.hog(imgs)
        logits = torch.nan_to_num(logits)
        p_x = self.softmax(logits)
        ids_selected = torch.multinomial(p_x, num_samples=len_selected_initialization, replacement=False)
        all_indices = torch.arange(p_x.shape[1]).repeat(p_x.shape[0], 1).to(ids_selected.device, non_blocking=True)
        selected_mask = torch.zeros_like(p_x, dtype=torch.bool)
        selected_mask.scatter_(1, ids_selected, True)
        unselected_mask = ~selected_mask
        ids_unselected = all_indices[unselected_mask].view(p_x.shape[0], p_x.shape[1] - ids_selected.shape[1])
        select_token = torch.gather(x, dim=1, index=ids_selected.unsqueeze(-1).repeat(1, 1, D))
        unselect_token = torch.gather(x, dim=1, index=ids_unselected.unsqueeze(-1).repeat(1, 1, D))
        (select_token, ids_selected), (unselect_token, ids_unselected) = self.token_select.token_expansion(
            select_token, ids_selected, unselect_token, ids_unselected, x)

        p_x_new = torch.gather(p_x, 1, ids_unselected)
        new_indices = torch.multinomial(p_x_new, num_samples=len_masked_reconstruct, replacement=False)
        final_ids_unselected = torch.gather(ids_unselected, 1, new_indices)

        mask = torch.zeros((x.shape[0], x.shape[1])).to(x.device, non_blocking=True)
        mask.scatter_(dim=-1, index=final_ids_unselected.long(), value=1)

        return select_token, mask, N, L, p_x

    def grid_patchify(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        return x

    def forward_encoder(self, x, mask_ratio, kept_mask_ratio):
        imgs = x
        x = self.grid_patchify(x)
        coords = None
        x, mask, N, L, p_x = self.random_masking(x, mask_ratio, kept_mask_ratio, imgs)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x_feats = []
        count = 0
        f = []
        if self.use_input:
            x_feats.append(x)
        for idx, blk in enumerate(self.blocks):
            count += 1
            x = blk(x)

        x = self.norm(x)
        return x, mask, N, L, coords, p_x

    def mask_tokens_grid(self, mask, N, L):
        N, L = N, L
        x = self.decoder_pos_embed[:, 1:].masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1,self.mask_token.shape[-1])
        x = x + self.mask_token
        return x

    def forward_decoder(self, y, mask, N, L):
        x = self.mask_tokens_grid(mask, N, L)
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, y)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward_loss_sampling(self, imgs, pred, mask):
        target = self.patchify(imgs)
        target = target.masked_select(mask.bool().unsqueeze(-1)).reshape(target.shape[0], -1, target.shape[-1])
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss_func = nn.MSELoss(reduction="none")
        loss_main = torch.mean(loss_func(pred, target), dim=-1)
        loss = loss_main.mean(dim=-1)
        return loss


    def forward(self, imgs, mask_ratio=0.75, kept_mask_ratio=0.5):
        with torch.cuda.amp.autocast():
            latent, mask, N, L, coords, p_x = self.forward_encoder(imgs, mask_ratio,kept_mask_ratio)
            pred = self.forward_decoder(latent, mask, N, L)
            loss_sampling = self.forward_loss_sampling(imgs, pred, mask)
            return loss_sampling



def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch16 = mae_vit_huge_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b