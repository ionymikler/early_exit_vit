# Adapted from Github's lucidrains/vit-pytorch/vit_pytorch/vit.py (3.Dec.24)

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class AttentionFeedForward(nn.Module):
    def __init__(self, embed_depth, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_depth),
            nn.Linear(embed_depth, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_depth),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, embed_depth, num_heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  num_heads
        project_out = not (num_heads == 1 and dim_head == embed_depth) # if more than one head, project out

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(embed_depth)
        

        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_depth, inner_dim * 3, bias = False) #QUESTION: Why bias=False?
        self.qkv_rearrage = Rearrange('b n (h d) -> b h n d', h = num_heads)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_depth),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        embed_dim = x.shape[-1]

        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        qkv_i = [self.to_qkv(x)[:, :, embed_dim * i:embed_dim * (i + 1)] for i in range(3)]
        q, k, v = map(self.qkv_rearrage, qkv_i)

        scaled_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.softmax(scaled_scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # TODO: Review this again. Understand the rearrange function
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, embed_depth, num_layers, num_attn_heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(embed_depth)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(
                    embed_depth = embed_depth,
                    num_heads = num_attn_heads,
                    dim_head = dim_head,
                    dropout = dropout
                ),
                
                AttentionFeedForward(
                    embed_depth=embed_depth,
                    hidden_dim=mlp_dim,
                    dropout = dropout
                )
            ]))

    def forward(self, x):
        _l_idx = 0
        for attn, ff in self.layers:
            print(f"[forward]: Layer {_l_idx}")
            x = attn(x) + x
            x = ff(x) + x
            _l_idx += 1
            
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size:int, patch_size:int, num_classes:int, embed_depth:int, num_layers_transformer:int, num_attn_heads:int, mlp_dim:int, pool:str = 'cls', channels:int = 3, dim_head:int = 64, dropout:float = 0., emb_dropout:float = 0.):
        # * to enforce only keyword arguments
        """
        Initializes the Vision Transformer (ViT) model.

        Args:
            image_size (int or tuple): Size of the input image. If an int is provided, it is assumed to be the size of both dimensions.
            patch_size (int or tuple): Size of the patches to be extracted from the input image. If an int is provided, it is assumed to be the size of both dimensions.
            num_classes (int): Number of output classes.
            embed_depth (int): Dimension of the embeddings.
            transformer_layers (int): Number of transformer layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the MLP (Feed-Forward) layer.
            pool (str, optional): Pooling type, either 'cls' (class token) or 'mean' (mean pooling). Default is 'cls'.
            channels (int, optional): Number of input channels. Default is 3.
            dim_head (int, optional): Depth dimension of the attention matrices in each head. Default is 64.
            dropout (float, optional): Dropout rate for the transformer. Default is 0.
            emb_dropout (float, optional): Dropout rate for the embedding layer. Default is 0.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_depth),
            nn.LayerNorm(embed_depth),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_depth))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_depth))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            embed_depth=embed_depth,
            num_layers=num_layers_transformer,
            num_attn_heads=num_attn_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout= dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(embed_depth, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # take cls token or average all tokens (pooling)

        x = self.to_latent(x) # identity, just for shape
        return self.mlp_head(x)
