import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads=8, embed_size=512):
        super().__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        self.fc_value = nn.Linear(self.head_dim, self.head_dim)
        self.fc_key = nn.Linear(self.head_dim, self.head_dim)
        self.fc_query = nn.Linear(self.head_dim, self.head_dim)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask):
        # query: N x seq_len x heads x head_dim
        N = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        value = self.fc_value(value)
        key = self.fc_key(key)
        query = self.fc_query(query)

        product = torch.einsum("nqhd, nkhd -> nhqk", [query, key])
        product = product.masked_fill(mask == 0, float(-1e9))
        # value: N x value_len x heads x head_dim 
        # -> N x ? x heads x head_dim
        out = torch.softmax(product / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql, nlhd -> nqhd", [out, value])
        out = out.reshape(N, query_len, self.embed_size)

        out = self.fc_out(out)

        return out
    
class Block(nn.Module): # contains 2 sublayers
    def __init__(self, embed_size, heads, dropout_p):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.attention = MultiHeadAttention(heads, embed_size)

    def forward(self, query, key, value, mask):
        x1 = self.attention(query, key, value, mask)
        x2 = self.dropout(self.norm1(x1 + query))
        out = self.feed_forward(x2)
        out = self.dropout(self.norm2(out + x2))

        return out
    
class Encoder(nn.Module):
    def __init__(self, embed_size, heads, dropout_p, source_vocab_len, device, num_layers=6, max_len=100): # max_len used for position encoding
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.device = device
        self.dropout = nn.Dropout(dropout_p)
        self.source_vocab_len = source_vocab_len
        self.num_layers = num_layers
        self.max_len = max_len

        self.word_embedding = nn.Embedding(source_vocab_len, embed_size)
        self.position_encoding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([Block(embed_size, heads, dropout_p) for _ in range(num_layers)])

    def forward(self, x, mask):
        N, seq_len = x.shape
        position = torch.arange(0, seq_len).repeat(N).reshape(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_encoding(position))

        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x
    
class Big_Block(nn.Module): # contains additional part compared to encoder
    def __init__(self, embed_size, heads, dropout_p, device):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = nn.Dropout(dropout_p)
        
        self.norm = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(heads, embed_size)
        self.block = Block(embed_size, heads, dropout_p)
        self.device = device

    def forward(self, x, key, value, src_mask, trg_mask):
        query = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(query + x))
        out = self.block(query, key, value, src_mask)

        return out
    
class Decoder(nn.Module):
    def __init__(self, embed_size, heads, dropout_p, target_vocab_size, device, num_layers=6, max_len=100):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = nn.Dropout(dropout_p)
        self.device = device 

        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_encoding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([Big_Block(embed_size, heads, dropout_p, device) for _ in range(num_layers)])

    def forward(self, x, encoder_out, src_mask, trg_mask):
        N, seq_len = x.shape
        position = torch.arange(0, seq_len).repeat(N).reshape(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_encoding(position))

        for layer in self.layers:
            out = layer(x, encoder_out, encoder_out, src_mask, trg_mask)

        return out
    
class Transformer(nn.Module):
    def __init__(self, source_vocab_len, target_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512, num_layers=6, heads=8, dropout_p=0.1, device='cuda', max_len=100):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.num_layers = num_layers
        self.device = device
        self.dropout_p = dropout_p
        self.max_len = max_len
        self.source_vocab_len = source_vocab_len
        self.target_vocab_size = target_vocab_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(embed_size, heads, dropout_p, source_vocab_len, device, num_layers, max_len)
        self.decoder = Decoder(embed_size, heads, dropout_p, target_vocab_size, device, num_layers, max_len)

        self.last_fc = nn.Linear(embed_size, target_vocab_size)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

    def forward(self, source, target):
        src_mask = self.make_src_mask(source)
        trg_mask = self.make_trg_mask(target)
        encoder_out = self.encoder(source, src_mask)
        out = self.decoder(target, encoder_out, src_mask, trg_mask)
        out = self.last_fc(out)

        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device='cpu').to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)