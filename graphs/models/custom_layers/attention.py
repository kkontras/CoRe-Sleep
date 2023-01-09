
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

class Channel_Seq_Small_Attention(nn.Module):
    def __init__(self, query_dim, modalities = 8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n m (h d) -> (b h) n m d', h = h), (q, k, v))

        sim = einsum('b i m d, b j z d -> b i j m z', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -3)

        out = einsum('b i j m z, b j m d -> b i z d', attn, v)

        out = rearrange(out, '(b h) n m d -> b n m (h d)', h = h)

        return self.to_out(out)
class Channel_Seq_Big_Attention(nn.Module):
    def __init__(self, query_dim, modalities = 8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim*modalities, query_dim*modalities),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        context = default(context, x)
        x_shape = x.shape

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n m (h d) -> (b h) n m d', h = h), (q, k, v))

        sim = einsum('b i m d, b j z d -> b i j m z', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -3)

        out = einsum('b i j m z, b j m d -> b i z d', attn, v)
        out = rearrange(out, '(b h) n m d -> b n (m h d)', h = h)

        return self.to_out(out).view(x_shape)
class Channel_Seq_ViProj_Small_Attention(nn.Module):
    def __init__(self, query_dim, modalities=8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.modalities = modalities

        self.pad = nn.ReflectionPad2d((2, 2, 0, 0))
        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=(1,5), bias=False)
        self.to_kv = nn.Conv2d(query_dim, inner_dim*2, kernel_size=(1,5), stride=(1,2),bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(self.pad(x.permute(0,3,2,1))).permute(0,3,2,1)

        context = default(context, x)

        k, v = self.to_kv(self.pad(context.permute(0,3,2,1))).permute(0,3,2,1).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n m (h d) -> (b h) n m d', h = h), (q, k, v))

        sim = einsum('b i m d, b j z d -> b i j m z', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -3)

        out = einsum('b i j m z, b j m d -> b i z d', attn, v)

        out = rearrange(out, '(b h) n m d -> b n m (h d)', h = h)

        return self.to_out(out)
class Channel_Seq_ViProj_Big_Attention(nn.Module):
    def __init__(self, query_dim, modalities = 8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.pad = nn.ReflectionPad2d((2, 2, 0, 0))
        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=(1,5), bias=False)
        self.to_kv = nn.Conv2d(query_dim, inner_dim*2, kernel_size=(1,5), stride=(1,2),bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim*modalities, query_dim*modalities),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        x_shape = x.shape
        context = default(context, x)

        q = self.to_q(self.pad(x.permute(0,3,2,1))).permute(0,3,2,1)
        k, v = self.to_kv(self.pad(context.permute(0,3,2,1))).permute(0,3,2,1).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n m (h d) -> (b h) n m d', h = h), (q, k, v))

        sim = einsum('b i m d, b j z d -> b i j m z', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -3)

        out = einsum('b i j m z, b j m d -> b i z d', attn, v)
        out = rearrange(out, '(b h) n m d -> b n (m h d)', h = h)
        out = self.to_out(out)

        return out.view(x_shape)

class Seq_Big_Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)

        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)
class Seq_Small_Attention(nn.Module):
    def __init__(self, query_dim, modalities = 8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.modalities = modalities

        self.to_q = nn.Linear(query_dim* modalities, inner_dim* modalities, bias = False)
        self.to_kv = nn.Linear(context_dim* modalities, inner_dim* modalities * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        m = self.modalities
        x_shape = x.shape
        q = self.to_q(x)
        context = default(context, x)

        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n (m d) -> b n m (h d)', h = h, m=m)

        return self.to_out(out).view(x_shape)
class Seq_ViProj_Big_Attention(nn.Module):
    def __init__(self, query_dim, modalities = 8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.pad = nn.ReflectionPad2d((2, 2, 0, 0))

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=(1,5), bias=False)
        self.to_kv = nn.Conv2d(query_dim, inner_dim*2, kernel_size=(1,5),bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim* modalities, query_dim* modalities),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        x_shape = x.shape

        q = self.to_q(self.pad(x.permute(0,3,2,1))).permute(0,3,2,1).flatten(start_dim=2)

        context = default(context, x)

        k, v = self.to_kv(self.pad(context.permute(0,3,2,1))).permute(0,3,2,1).chunk(2, dim = -1)
        k = k.flatten(start_dim=2)
        v = v.flatten(start_dim=2)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        return self.to_out(out).view(x_shape)
class Seq_ViProj_Small_Attention(nn.Module):
    def __init__(self, query_dim, modalities = 8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.modalities = modalities

        self.pad = nn.ReflectionPad2d((2, 2, 0, 0))

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=(1,5), bias=False)
        self.to_kv = nn.Conv2d(query_dim, inner_dim*2, kernel_size=(1,5),bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        m = self.modalities
        x_shape = x.shape

        q = self.to_q(self.pad(x.permute(0,3,2,1))).permute(0,3,2,1).flatten(start_dim=2)

        context = default(context, x)

        k, v = self.to_kv(self.pad(context.permute(0,3,2,1))).permute(0,3,2,1).chunk(2, dim = -1)
        k = k.flatten(start_dim=2)
        v = v.flatten(start_dim=2)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n (m d) -> b n m (h d)', h = h, m = m)

        return self.to_out(out).view(x_shape)

class Seq_CNN_Big_Attention(nn.Module):
    def __init__(self, query_dim, modalities = 8, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.modalities = modalities

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.pad = nn.ReflectionPad2d((2, 2, 0, 0))

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=(1,5), bias=False)
        self.to_kv = nn.Conv2d(query_dim, inner_dim*2, kernel_size=(1,5), bias=False)

        self.to_out = nn.Conv2d(inner_dim, query_dim*modalities, kernel_size=(modalities,5), bias=False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim* modalities, query_dim* modalities),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x, context = None, mask = None):
        h = self.heads
        m = self.modalities

        x_shape = x.shape

        q = self.to_q(self.pad(x.permute(0,3,2,1))).permute(0,3,2,1)

        context = default(context, x)

        k, v = self.to_kv(self.pad(context.permute(0,3,2,1))).permute(0,3,2,1).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n m (h d) -> (b h) n (m d)', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n (m d) -> b (h d) m n', h = h, m=m)
        out = self.to_out(self.pad(out))
        out = rearrange(out, 'b (f m) c n -> b f m c n', m=m ).squeeze()
        return  out.view(x_shape)

def default(val, d):
    return val if exists(val) else d
def exists(val):
    return val is not None
class PreNorm(nn.Module):
    def __init__(self, dim, context_dim = None):
        super().__init__()
        # self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x):
        x = self.norm(x)

        # if exists(self.norm_context):
        #     context = kwargs['context']
        #     normed_context = self.norm_context(context)
        #     kwargs.update(context = normed_context)

        return x
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)



class My_Transformer_Layer_Ch_Small_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.1):
        super().__init__()
        self.attention1  = Channel_Seq_Small_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_Big_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.1):
        super().__init__()
        self.attention1  = Channel_Seq_Big_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_Small_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.15):
        super().__init__()

        self.attention1  = Channel_Seq_Small_Attention(query_dim = dmodel, dropout = dropout)
        self.norm1 = PreNorm(dmodel*modalities,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel*modalities, dropout = dropout)
        self.normff1 = PreNorm(dmodel*modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x.flatten(start_dim=2, end_dim=3))
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_Big_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.15):
        super().__init__()

        self.attention1  = Channel_Seq_Small_Attention(query_dim = dmodel, dropout = dropout)
        self.norm1 = PreNorm(dmodel*modalities,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel*modalities, dropout = dropout)
        self.normff1 = PreNorm(dmodel*modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x.flatten(start_dim=2, end_dim=3))
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_Small_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8,  dropout=0.1):
        super().__init__()
        self.modalities = modalities
        self.attention1 = Channel_Seq_Small_Attention(query_dim=dmodel, modalities = modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = x + self.dropout(x_att)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_Big_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8,  dropout=0.1):
        super().__init__()
        self.modalities = modalities
        self.attention1 = Channel_Seq_Big_Attention(query_dim=dmodel, modalities = modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = x + self.dropout(x_att)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_ViProj_Small_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.15):
        super().__init__()
        self.attention1  = Channel_Seq_ViProj_Small_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_ViProj_Big_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.15):
        super().__init__()
        self.attention1  = Channel_Seq_ViProj_Big_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_ViProj_Big_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout=0.15):
        super().__init__()

        self.attention1 = Channel_Seq_ViProj_Big_Attention(query_dim=dmodel, modalities= modalities, dropout=dropout)
        self.norm1 = PreNorm(dmodel * modalities, context_dim=dmodel)
        self.ff1 = FeedForward(dmodel * modalities, dropout=dropout)
        self.normff1 = PreNorm(dmodel * modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x.flatten(start_dim=2, end_dim=3))
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_ViProj_Small_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout=0.15):
        super().__init__()

        self.attention1 = Channel_Seq_ViProj_Small_Attention(query_dim=dmodel, modalities= modalities, dropout=dropout)
        self.norm1 = PreNorm(dmodel * modalities, context_dim=dmodel)
        self.ff1 = FeedForward(dmodel * modalities, dropout=dropout)
        self.normff1 = PreNorm(dmodel * modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x.flatten(start_dim=2, end_dim=3))
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_ViProj_Small_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8,  dropout=0.1):
        super().__init__()
        self.modalities = modalities
        self.attention1 = Channel_Seq_ViProj_Small_Attention(query_dim=dmodel, modalities = modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = x + self.dropout(x_att)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)
class My_Transformer_Layer_Ch_ViProj_Big_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8,  dropout=0.1):
        super().__init__()
        self.modalities = modalities
        self.attention1 = Channel_Seq_ViProj_Big_Attention(query_dim=dmodel, modalities = modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = x + self.dropout(x_att)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)

class My_Transformer_Layer_Big_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities = 8, dropout = 0.1):
        super().__init__()
        self.attention1  = Seq_Big_Attention(query_dim = dmodel* modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        x_inter_shape = x.shape
        x = x.flatten(start_dim=2)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = x.view(x_inter_shape)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Big_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.25):
        super().__init__()

        self.attention1  = Seq_Big_Attention(query_dim = dmodel*modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel*modalities,  context_dim = dmodel*modalities)
        self.ff1  = FeedForward(dmodel*modalities, dropout = dropout)
        self.normff1 = PreNorm(dmodel*modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1).flatten(start_dim=2,end_dim=3)
        context = default(context, x)
        x = self.attention1(x, context)
        x = x + self.dropout(x)
        x = self.norm1(x)
        x = self.ff1(x)
        x = x + self.dropout(x)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Big_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8,  dropout=0.1):
        super().__init__()
        self.modalities = modalities
        self.attention1 = Seq_Big_Attention(query_dim=dmodel*modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        x_inter_shape = x.shape
        x = x.flatten(start_dim=2,end_dim=3)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = (x + self.dropout(x_att)).view(x_inter_shape)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)
class My_Transformer_Layer_Small_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities = 8, dropout = 0.1):
        super().__init__()
        self.attention1  = Seq_Small_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        x_inter_shape = x.shape
        x = x.flatten(start_dim=2)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = x.view(x_inter_shape)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Small_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.25):
        super().__init__()

        self.attention1  = Seq_Small_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel*modalities,  context_dim = dmodel*modalities)
        self.ff1  = FeedForward(dmodel*modalities, dropout = dropout)
        self.normff1 = PreNorm(dmodel*modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1).flatten(start_dim=2,end_dim=3)
        context = default(context, x)
        x = self.attention1(x, context)
        x = x + self.dropout(x)
        x = self.norm1(x)
        x = self.ff1(x)
        x = x + self.dropout(x)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_Small_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8,  dropout=0.1):
        super().__init__()
        self.modalities = modalities
        self.attention1 = Seq_Small_Attention(query_dim=dmodel, modalities=modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        x_inter_shape = x.shape
        x = x.flatten(start_dim=1,end_dim=2)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = (x + self.dropout(x_att)).view(x_inter_shape)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)
class My_Transformer_Layer_ViProj_Small_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities= 8, dropout = 0.1):
        super().__init__()
        self.attention1  = Seq_ViProj_Small_Attention(query_dim = dmodel, modalities = modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_ViProj_Big_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities= 8, dropout = 0.1):
        super().__init__()
        self.attention1  = Seq_ViProj_Big_Attention(query_dim = dmodel, modalities = modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_ViProj_Small_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.15):
        super().__init__()

        self.attention1  = Seq_ViProj_Small_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel*modalities,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel*modalities, dropout = dropout)
        self.normff1 = PreNorm(dmodel*modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)

        x = x + self.dropout(xatt)
        x = self.norm1(x.flatten(start_dim=2, end_dim=3))
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_ViProj_Big_BigFF(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout = 0.15):
        super().__init__()

        self.attention1  = Seq_ViProj_Big_Attention(query_dim = dmodel, modalities=modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel*modalities,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel*modalities, dropout = dropout)
        self.normff1 = PreNorm(dmodel*modalities)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)

        x = x + self.dropout(xatt)
        x = self.norm1(x.flatten(start_dim=2, end_dim=3))
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
class My_Transformer_Layer_ViProj_Big_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout=0.1):
        super().__init__()
        self.modalities= modalities
        self.attention1 = Seq_ViProj_Big_Attention(query_dim=dmodel, modalities=modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = x + self.dropout(x_att)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)
class My_Transformer_Layer_ViProj_Small_8nets(nn.Module):
    def __init__(self, dmodel, modalities=8, dropout=0.1):
        super().__init__()
        self.modalities= modalities
        self.attention1 = Seq_ViProj_Small_Attention(query_dim=dmodel, modalities=modalities, dropout=dropout)
        for i in range(modalities):
            setattr(self, "norm%d" % i, PreNorm(dmodel, context_dim=dmodel))
            setattr(self, "ff%d" % i, FeedForward(dmodel, dropout=dropout))
            setattr(self, "normff%d" % i, PreNorm(dmodel))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        x_att = self.attention1(x, context)
        x = x + self.dropout(x_att)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "norm%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "ff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        xff = torch.cat(clf_outputs, dim=2)
        x = x + self.dropout(xff)
        clf_outputs = []
        for i in range(self.modalities):
            clf_outputs.append(getattr(self, "normff%d" % i)(x[:, :, i]).unsqueeze(dim=2))
        x = torch.cat(clf_outputs, dim=2)
        return x.view(x_shape)

class My_Transformer_Layer_CNN_Big_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities= 8, dropout = 0.1):
        super().__init__()
        self.attention1  = Seq_CNN_Big_Attention(query_dim = dmodel, modalities = modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        # self.ff1  = FeedForward(dmodel, dropout = dropout)
        # self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = self.norm1(x)
        # xff = self.ff1(x)
        # x = x + self.dropout(xff)
        # x = self.normff1(x)
        return x.view(x_shape)


class My_Transformers_SEDF_Big_SmallFF(nn.Module):
    def __init__(self, dmodel, modalities = 2, dropout = 0.1):
        super().__init__()
        self.attention1  = Seq_Big_Attention(query_dim = dmodel* modalities, dropout = dropout)
        self.norm1 = PreNorm(dmodel,  context_dim = dmodel)
        self.ff1  = FeedForward(dmodel, dropout = dropout)
        self.normff1 = PreNorm(dmodel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None):
        #x must be batch x features x modalities x seq
        x_shape = x.shape
        x = x.permute(0,3,2,1)
        x_inter_shape = x.shape
        x = x.flatten(start_dim=2)
        context = default(context, x)
        xatt = self.attention1(x, context)
        x = x + self.dropout(xatt)
        x = x.view(x_inter_shape)
        x = self.norm1(x)
        xff = self.ff1(x)
        x = x + self.dropout(xff)
        x = self.normff1(x)
        return x.view(x_shape)
