import math
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
import json
from collections import Counter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=256, dropout=0.2, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)

        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        return self.fc_out(output)

def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def generate_padding_mask(src: Tensor, pad_idx=0) -> Tensor:
    return (src == pad_idx)

def evaluate(model, data_loader, loss_fn, vocab_size, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch[0].to(device)
            src = batch_data[:, :-1].transpose(0,1)
            tgt_input = batch_data[:, :-1].transpose(0,1)
            tgt_output = batch_data[:, 1:].transpose(0,1)

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            src_key_padding_mask = generate_padding_mask(batch_data[:, :-1]).to(device)
            tgt_key_padding_mask = generate_padding_mask(batch_data[:, :-1]).to(device)

            output = model(src, tgt_input, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)

            loss = loss_fn(output.view(-1, vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Top-p Sampling：從最大機率開始累加，直到總和超過p（例0.9），只在這些token中進行抽樣
def sample_next_token(logits, p=0.9):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    p_tensor = torch.tensor(p, device=logits.device)

    cutoff_index = torch.searchsorted(cumulative_probs, p_tensor, right=False).item()
    filtered_probs = sorted_probs[..., :cutoff_index + 1]
    filtered_indices = sorted_indices[..., :cutoff_index + 1]

    filtered_probs = filtered_probs / torch.sum(filtered_probs)
    next_token = torch.multinomial(filtered_probs, 1)

    return filtered_indices[..., next_token].item()

def generate_sentence(model, vocab, idx2word, device, max_len: int = 30):
    model.eval()
    with torch.no_grad():
        valid_words = [w for w in vocab.keys() if w not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>")]
        start_word = random.choice(valid_words)
        input_ids = [vocab[start_word]]

        src_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        src_key_padding_mask = generate_padding_mask(src_ids, pad_idx=vocab["<PAD>"]).to(device)
        src = src_ids.transpose(0, 1)

        tgt_input = torch.tensor([[vocab["<SOS>"]]], dtype=torch.long).to(device)

        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            tgt_key_padding_mask = generate_padding_mask(tgt_input.transpose(0, 1), pad_idx=vocab["<PAD>"]).to(device)

            output = model(src, tgt_input, src_mask=None, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)

            next_token = sample_next_token(output[-1, 0, :], p=0.9)
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]], device=device)], dim=0)
            if next_token == vocab["<EOS>"]:
                break

        output_ids = tgt_input.squeeze(1).tolist()
        return ''.join([idx2word[idx] for idx in output_ids if idx2word[idx] not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>")])

with open('tokenized_data.json', 'r', encoding='utf-8-sig') as f:
    sentences = json.load(f)

word_freq = Counter(w for s in sentences for w in s)

vocab = {"<PAD>": 0,  "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
for word, freq in word_freq.items():
    if freq >= 5:
        vocab[word] = len(vocab)
vocab_size = len(vocab)

indexed_sentences = []
for sentence in sentences:
    indexed_sentence = [vocab["<SOS>"]] + [vocab.get(word, vocab["<UNK>"]) for word in sentence] + [vocab["<EOS>"]]
    indexed_sentences.append(indexed_sentence)

max_len = max(len(s) for s in indexed_sentences)
for i in range(len(indexed_sentences)):
    while len(indexed_sentences[i]) < max_len:
        indexed_sentences[i].append(vocab["<PAD>"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_sentences, val_sentences = train_test_split(indexed_sentences, test_size=0.2, random_state=42)

train_tensor = torch.tensor(train_sentences, dtype=torch.long)
val_tensor = torch.tensor(val_sentences, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size)

model = TransformerSeq2Seq(vocab_size, pad_idx=vocab["<PAD>"]).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"], label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch_data = batch[0].to(device)

        src = batch_data[:, :-1].transpose(0, 1)
        tgt = batch_data[:, 1:].transpose(0, 1)

        tgt_input = batch_data[:, :-1].transpose(0, 1)
        tgt_output = batch_data[:, 1:].transpose(0, 1)

        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        src_key_padding_mask = generate_padding_mask(src.transpose(0, 1)).to(device)
        tgt_key_padding_mask = generate_padding_mask(tgt_input.transpose(0, 1)).to(device)
            
        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask)

        loss = loss_fn(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, loss_fn, vocab_size, device)
    scheduler.step(val_loss)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

idx2word = {idx: word for word, idx in vocab.items()}

for i in range(5):
    print(f"第{i+1}句：", generate_sentence(model, vocab, idx2word, device))