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

# Embedding (vocab_size → d_model)
#    ↓
# PositionalEncoding (加上位置資訊)
#    ↓
# TransformerEncoderLayer × N (num_layers)層：
# 將encoder_layer區塊複製堆疊 num_layers 次
#     └─ MultiHeadAttention (input/output: d_model)
#     └─ FeedForward (d_model → dim_feedforward → d_model)
#    ↓
# Linear (d_model → vocab_size)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)

        self._init_weights()

    # 初始化Embedding層 及 輸出Linear層的權重
    # 不包含Transformer Encoder Layer 的權重(由 PyTorch 預設自動初始化)
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.linear(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def generate_padding_mask(src: Tensor, pad_idx=0) -> Tensor:
    return (src.transpose(0,1) == pad_idx)

def evaluate(model, data_loader, loss_fn, vocab_size, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch[0].to(device)
            inputs = batch_data[:, :-1].transpose(0,1)
            targets = batch_data[:, 1:].transpose(0,1)

            src_mask = generate_square_subsequent_mask(inputs.size(0)).to(device)
            src_key_padding_mask = generate_padding_mask(inputs).to(device)
            output = model(inputs, src_mask, src_key_padding_mask)
            loss = loss_fn(output.view(-1, vocab_size), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

def sample_next_token(logits):
    probs = torch.softmax(logits, dim=-1)
    # 根據機率分布隨機抽樣一個 token（這是隨機取樣，而非最大機率選擇）
    next_token = torch.multinomial(probs, 1)
    return next_token.item()

def generate_sentence(model, vocab, idx2word, device, max_len):
    model.eval()
    with torch.no_grad():
        valid_words = [w for w in vocab.keys() if w not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>")]
        start_word = random.choice(valid_words)
        input_ids = [vocab[start_word]]

        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)

        for _ in range(max_len - 1):
            src_mask = generate_square_subsequent_mask(input_tensor.size(0)).to(device)
            src_key_padding_mask = generate_padding_mask(input_tensor).to(device)
            output = model(input_tensor, src_mask, src_key_padding_mask)
            logits = output[-1, 0, :]
            next_token_id = sample_next_token(logits)

            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=0)

            if next_token_id == vocab["<EOS>"]:
                break

        output_ids = input_tensor.squeeze(1).tolist()
        return ''.join([idx2word[idx] for idx in output_ids if idx2word[idx] not in ("<PAD>", "<SOS>", "<EOS>", "<UNK>")])

with open('tokenized_data.json', 'r', encoding='utf-8-sig') as f:
    sentences = json.load(f)

word_freq = Counter(w for s in sentences for w in s)

vocab = {"<PAD>": 0,  "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
for word, freq in word_freq.items():
    if freq >= 10:
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

model = TransformerModel(vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"], label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch_data = batch[0].to(device)
        inputs = batch_data[:, :-1].transpose(0,1)
        targets = batch_data[:, 1:].transpose(0,1)

        src_mask = generate_square_subsequent_mask(inputs.size(0)).to(device)
        src_key_padding_mask = generate_padding_mask(inputs).to(device)
        
        optimizer.zero_grad()
        output = model(inputs, src_mask, src_key_padding_mask)
        loss = loss_fn(output.view(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, loss_fn, vocab_size, device)
    scheduler.step(val_loss)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

idx2word = {idx: word for word, idx in vocab.items()}

for i in range(5):
    print(f"第{i+1}句：", generate_sentence(model, vocab, idx2word, device, max_len=30))