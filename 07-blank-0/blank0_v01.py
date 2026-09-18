import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
DATA_FILE = 'blank_dataset_v1_tokens.bin'
METADATA_FILE = 'blank_dataset_v1_tokens_meta.json'
TOKENIZER_FILE = 'blank_tokenizer_v1.json'
BEST_CHECKPOINT = 'blank0_v01_best.pt'
LATEST_CHECKPOINT = 'blank0_v01_latest.pt'
FINAL_CHECKPOINT = 'blank0_v01_final.pt'
for filename in [DATA_FILE, METADATA_FILE, TOKENIZER_FILE]:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
vocab_size = metadata['vocab_size']
total_tokens = metadata['total_tokens']
train_tokens = metadata['train_tokens']
validation_tokens = metadata['validation_tokens']
print('Total tokens:', f'{total_tokens:,}')
print('Training tokens:', f'{train_tokens:,}')
print('Validation tokens:', f'{validation_tokens:,}')
print('Vocabulary:', f'{vocab_size:,}')
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
if not torch.cuda.is_available():
    raise RuntimeError('ROCm GPU is unavailable.')
device = torch.device('cuda:0')
print('\nGPU:', torch.cuda.get_device_name(0))
print('HIP:', torch.version.hip)
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2), 'GB')
torch.manual_seed(42)
context_length = 512
batch_size = 16
d_model = 384
number_of_layers = 8
number_of_heads = 6
head_size = d_model // number_of_heads
assert d_model % number_of_heads == 0
assert head_size % 2 == 0
mlp_hidden = 1024
training_steps = 8000
maximum_learning_rate = 0.0003
minimum_learning_rate = 3e-05
warmup_steps = 300
weight_decay = 0.1
gradient_clip = 1.0
evaluation_interval = 250
evaluation_batches = 20
RESUME = True
print('\nBLANK-0 v0.1')
print('Context:', context_length)
print('Batch size:', batch_size)
print('Width:', d_model)
print('Layers:', number_of_layers)
print('Heads:', number_of_heads)
print('Head dimension:', head_size)
print('SwiGLU hidden:', mlp_hidden)
print('Training steps:', training_steps)
print('\nLoading binary token dataset...')
binary_tokens = np.memmap(DATA_FILE, dtype=np.uint16, mode='r')
if len(binary_tokens) != total_tokens:
    raise RuntimeError('Token count does not match metadata.')
cpu_tokens = torch.from_numpy(np.asarray(binary_tokens, dtype=np.int64))
all_tokens = cpu_tokens.to(device)
del cpu_tokens
del binary_tokens
train_data = all_tokens[:train_tokens]
validation_data = all_tokens[train_tokens:]
print('Dataset loaded onto GPU.')
batch_offsets = torch.arange(context_length, device=device)

def get_batch(split):
    if split == 'train':
        source = train_data
    elif split == 'validation':
        source = validation_data
    else:
        raise ValueError(split)
    starts = torch.randint(low=0, high=len(source) - context_length - 1, size=(batch_size,), device=device)
    positions = starts[:, None] + batch_offsets[None, :]
    X = source[positions]
    Y = source[positions + 1]
    return (X, Y)

class RMSNorm(nn.Module):

    def __init__(self, dimension, epsilon=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, x):
        rms = torch.mean(x * x, dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(rms + self.epsilon)
        return normalized * self.weight

class RotaryEmbedding(nn.Module):

    def __init__(self, dimension, maximum_length, base=10000.0):
        super().__init__()
        inverse_frequency = 1.0 / base ** (torch.arange(0, dimension, 2).float() / dimension)
        positions = torch.arange(maximum_length).float()
        frequencies = torch.outer(positions, inverse_frequency)
        cosine = frequencies.cos()
        sine = frequencies.sin()
        self.register_buffer('cosine', cosine, persistent=False)
        self.register_buffer('sine', sine, persistent=False)

    def forward(self, Q, K):
        T = Q.shape[-2]
        cosine = self.cosine[:T].to(dtype=Q.dtype)
        sine = self.sine[:T].to(dtype=Q.dtype)
        cosine = cosine[None, None, :, :]
        sine = sine[None, None, :, :]
        Q = apply_rotary(Q, cosine, sine)
        K = apply_rotary(K, cosine, sine)
        return (Q, K)

def apply_rotary(x, cosine, sine):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated.flatten(-2)

class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryEmbedding(head_size, context_length)
        causal_mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.reshape(B, T, number_of_heads, head_size).transpose(1, 2)
        K = K.reshape(B, T, number_of_heads, head_size).transpose(1, 2)
        V = V.reshape(B, T, number_of_heads, head_size).transpose(1, 2)
        Q, K = self.rotary(Q, K)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(head_size)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attention = F.softmax(scores, dim=-1)
        out = attention @ V
        out = out.transpose(1, 2).contiguous().reshape(B, T, d_model)
        return self.output(out)

class SwiGLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(d_model, mlp_hidden, bias=False)
        self.up = nn.Linear(d_model, mlp_hidden, bias=False)
        self.down = nn.Linear(mlp_hidden, d_model, bias=False)

    def forward(self, x):
        gated = F.silu(self.gate(x)) * self.up(x)
        return self.down(gated)

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = CausalSelfAttention()
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU()

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class BlankZero(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(number_of_layers)])
        self.final_norm = RMSNorm(d_model)
        self.apply(self._initialize_weights)
        residual_scale = 1.0 / math.sqrt(2 * number_of_layers)
        for block in self.blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.mlp.down.weight.data.mul_(residual_scale)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, X, targets=None):
        B, T = X.shape
        if T > context_length:
            raise ValueError('Input exceeds context window.')
        x = self.token_embedding(X)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        return (logits, loss)
model = BlankZero().to(device)
parameter_count = sum((parameter.numel() for parameter in model.parameters()))
print('\nTRAINABLE PARAMETERS:', f'{parameter_count:,}')
optimizer = torch.optim.AdamW(model.parameters(), lr=maximum_learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)

def get_learning_rate(step):
    if step < warmup_steps:
        return maximum_learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (training_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + cosine * (maximum_learning_rate - minimum_learning_rate)

@torch.no_grad()
def evaluate():
    model.eval()
    results = {}
    for split in ['train', 'validation']:
        losses = []
        for _ in range(evaluation_batches):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())
        average_loss = sum(losses) / len(losses)
        perplexity = math.exp(min(average_loss, 20))
        results[split] = {'loss': average_loss, 'perplexity': perplexity}
    model.train()
    return results

def save_checkpoint(filename, step, best_validation_loss):
    torch.save({'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_validation_loss': best_validation_loss, 'config': {'vocab_size': vocab_size, 'context_length': context_length, 'd_model': d_model, 'number_of_layers': number_of_layers, 'number_of_heads': number_of_heads, 'head_size': head_size, 'mlp_hidden': mlp_hidden}}, filename)
start_step = 0
best_validation_loss = float('inf')
if RESUME and os.path.exists(LATEST_CHECKPOINT):
    print('\nLoading latest checkpoint...')
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    best_validation_loss = checkpoint['best_validation_loss']
    print('Resuming from step:', start_step)
print('\nTRAINING BLANK-0\n')
model.train()
last_report_time = time.perf_counter()
steps_since_report = 0
training_start = time.perf_counter()
for step in range(start_step, training_steps):
    if step % evaluation_interval == 0:
        torch.cuda.synchronize()
        now = time.perf_counter()
        if steps_since_report > 0:
            training_tokens_processed = steps_since_report * batch_size * context_length
            throughput = training_tokens_processed / (now - last_report_time)
        else:
            throughput = 0.0
        metrics = evaluate()
        train_loss = metrics['train']['loss']
        validation_loss = metrics['validation']['loss']
        validation_perplexity = metrics['validation']['perplexity']
        learning_rate = get_learning_rate(step)
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f'step={step:5d} | train={train_loss:.4f} | val={validation_loss:.4f} | ppl={validation_perplexity:.2f} | lr={learning_rate:.2e} | tok/s={throughput:,.0f} | VRAM={reserved:.2f} GB')
        save_checkpoint(LATEST_CHECKPOINT, step, best_validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(BEST_CHECKPOINT, step, best_validation_loss)
            print('  -> new best checkpoint')
        torch.cuda.synchronize()
        last_report_time = time.perf_counter()
        steps_since_report = 0
    learning_rate = get_learning_rate(step)
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = learning_rate
    X, Y = get_batch('train')
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    steps_since_report += 1
torch.cuda.synchronize()
training_end = time.perf_counter()
final_metrics = evaluate()
print('\nFINAL RESULTS')
print('Training loss:', round(final_metrics['train']['loss'], 4))
print('Validation loss:', round(final_metrics['validation']['loss'], 4))
print('Validation perplexity:', round(final_metrics['validation']['perplexity'], 2))
print('Best validation loss:', round(best_validation_loss, 4))
print('Current-session time:', round(training_end - training_start, 2), 'seconds')
print('GPU reserved:', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 2), 'GB')
save_checkpoint(FINAL_CHECKPOINT, training_steps, best_validation_loss)
print('\nFinal checkpoint saved:', FINAL_CHECKPOINT)
if os.path.exists(BEST_CHECKPOINT):
    checkpoint = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded best model from step:', checkpoint['step'])

@torch.no_grad()
def generate(prompt, number_of_tokens=300, temperature=0.8, top_k=50):
    model.eval()
    prompt_ids = tokenizer.encode(prompt).ids
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(number_of_tokens):
        context = generated[:, -context_length:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k is not None and top_k < vocab_size:
            values, _ = torch.topk(logits, top_k)
            threshold = values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return tokenizer.decode(generated[0].tolist())
print('\n========================================')
print('GENERATED SAMPLE 1')
print('========================================\n')
print(generate('Artificial intelligence is', number_of_tokens=250, temperature=0.8, top_k=50))
print('\n========================================')
print('GENERATED SAMPLE 2')
print('========================================\n')
print(generate('The history of Europe', number_of_tokens=250, temperature=0.8, top_k=50))
