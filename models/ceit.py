import torch
from torch import nn, einsum
import torch.nn.functional as F
from mask import RelationshipLayer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=14, w=14)
            )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerLeFF(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, scale=4, depth_kernel=3, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, LeFF(dim, scale, depth_kernel)))
            ]))

    def forward(self, x):
        c = list()
        for attn, leff in self.layers:
            x = attn(x)
            cls_tokens = x[:, 0]
            c.append(cls_tokens)
            x = leff(x[:, 1:])
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        return x, torch.stack(c).transpose(0, 1)


class LCA(nn.Module):
    # I remove Residual connection from here, in paper author didn't explicitly mentioned to use Residual connection,
    # so I removed it, althougth with Residual connection also this code will work.
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
            PreNorm(dim, LCAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x[:, -1].unsqueeze(1)

            x = x[:, -1].unsqueeze(1) + ff(x)
        return x


class CeiT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim=192, depth=12, heads=3, pool='cls', in_channels=3,
                 out_channels=32, dim_head=64, dropout=0.,
                 emb_dropout=0., conv_kernel=7, stride=2, depth_kernel=3, pool_kernel=3, scale_dim=4, with_lca=False,
                 lca_heads=4, lca_dim_head=48, lca_mlp_dim=384):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # IoT
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel, stride, 4),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_kernel, stride)
        )

        feature_size = image_size // 4

        assert feature_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size // patch_size) ** 2
        patch_dim = out_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerLeFF(dim, depth, heads, dim_head, scale_dim, depth_kernel, dropout)

        self.with_lca = with_lca
        if with_lca:
            self.LCA = LCA(dim, lca_heads, lca_dim_head, lca_mlp_dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.relationship_layer = RelationshipLayer(similarity='cos')

    def forward(self, img, state):
        relation_lists = []  # 6 + 1

        # After conv layer
        x = self.conv(img)
        if state[0]:
            r1 = self.relationship_layer(x, state[0])
            relation_lists.append(r1)

        # After patch embedding
        x = self.to_patch_embedding(x)
        if state[1]:
            r2 = self.relationship_layer(x, state[1])
            relation_lists.append(r2)

        b, n, _ = x.shape

        # After adding cls token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # After adding positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        if state[2]:
            r3 = self.relationship_layer(x, state[2])
            relation_lists.append(r3)

        x = self.dropout(x)

        # After transformer
        x, c = self.transformer(x)
        if state[3]:
            r4 = self.relationship_layer(x, state[3])
            relation_lists.append(r4)

        # After LCA or pooling
        if self.with_lca:
            lca_output = self.LCA(c)
            if state[4]:
                r5 = self.relationship_layer(x, state[4])
                relation_lists.append(r5)
            x = lca_output[:, 0]
        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            if state[4]:
                r5 = self.relationship_layer(x, state[4])
                relation_lists.append(r5)

        # After to_latent
        x = self.to_latent(x)
        if state[5]:
            r6 = self.relationship_layer(x, state[5])
            relation_lists.append(r6)

        # After mlp_head with softmax
        output = self.mlp_head(x)
        if state[6]:
            output_softmax = F.softmax(output, dim=1)
            r7 = self.relationship_layer(output_softmax, state[6])
            relation_lists.append(r7)

        return output, relation_lists

def ceit_t(num_classes=200):
    """CeiT-B (Base) model

    Modified according to specifications
    """
    return CeiT(
        image_size=224,
        patch_size=4,
        num_classes=num_classes,
        dim=192,  # Embedding dimension
        depth=12,  # Number of transformer blocks
        heads=3,  # Number of attention heads
        in_channels=3,  # RGB images
        out_channels=32,  # Output channels after conv layer
        dim_head=64,  # Dimension of each attention head
        dropout=0.,
        emb_dropout=0.,
        conv_kernel=7,  # Kernel size for IoT conv
        stride=2,  # Stride for IoT conv
        depth_kernel=3,  # Kernel size for depth-wise conv in LeFF
        pool_kernel=2,  # Kernel size for pooling in IoT
        scale_dim=4,  # Scale dimension for LeFF
        with_lca=True,  # Use LCA
        lca_heads=4,  # Number of LCA heads
        lca_dim_head=48,  # Dimension of each LCA head
        lca_mlp_dim=384  # MLP dimension in LCA
    )