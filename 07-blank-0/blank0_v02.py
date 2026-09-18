import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
DATA_FILE = 'blank_dataset_v2_tokens.bin'
METADATA_FILE = 'blank_dataset_v2_tokens_meta.json'
TOKENIZER_FILE = 'blank_tokenizer_v1.json'
BEST_CHECKPOINT = 'blank0_v02_best.pt'
LATEST_CHECKPOINT = 'blank0_v02_latest.pt'
FINAL_CHECKPOINT = 'blank0_v02_final.pt'
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
torch.cuda.manual_seed_all(42)
context_length = 512
batch_size = 16
d_model = 384
number_of_layers = 8
number_of_heads = 6
head_size = d_model // number_of_heads
assert d_model % number_of_heads == 0
assert head_size % 2 == 0
mlp_hidden = 1024
training_steps = 34880
maximum_learning_rate = 0.0003
minimum_learning_rate = 3e-05
warmup_steps = 1300
weight_decay = 0.1
gradient_clip = 1.0
evaluation_interval = 1000
evaluation_batches = 30
RESUME = True
tokens_per_update = batch_size * context_length
dataset_equivalent_steps = math.ceil(train_tokens / tokens_per_update)
print('\nBLANK-0 v0.2')
print('Context:', context_length)
print('Batch size:', batch_size)
print('Tokens/update:', f'{tokens_per_update:,}')
print('Width:', d_model)
print('Layers:', number_of_layers)
print('Heads:', number_of_heads)
print('Head dimension:', head_size)
print('SwiGLU hidden:', mlp_hidden)
print('Training updates:', f'{training_steps:,}')
print('Dataset-equivalent updates:', f'{dataset_equivalent_steps:,}')
print('Planned token predictions:', f'{training_steps * tokens_per_update:,}')
print('\nLoading binary token dataset...')
binary_tokens = np.memmap(DATA_FILE, dtype=np.uint16, mode='r')
if len(binary_tokens) != total_tokens:
    raise RuntimeError('Token count does not match metadata.')
print('Converting token IDs to int64...')
cpu_array = np.asarray(binary_tokens, dtype=np.int64)
cpu_tokens = torch.from_numpy(cpu_array)
print('Moving token dataset to GPU...')
all_tokens = cpu_tokens.to(device)
del cpu_tokens
del cpu_array
del binary_tokens
train_data = all_tokens[:train_tokens]
validation_data = all_tokens[train_tokens:]
print('Dataset loaded onto GPU.')
dataset_memory_gb = all_tokens.numel() * all_tokens.element_size() / 1024 ** 3
print('GPU dataset memory:', round(dataset_memory_gb, 2), 'GB')
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
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.epsilon)
        return normalized * self.weight

def apply_rotary(x, cosine, sine):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated.flatten(-2)

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
parameter_count = sum((p.numel() for p in model.parameters()))
print('\nTRAINABLE PARAMETERS:', f'{parameter_count:,}')
expected_parameters = 15735168
if parameter_count != expected_parameters:
    print('WARNING: parameter count differs from Blank-0 v0.1.')
optimizer = torch.optim.AdamW(model.parameters(), lr=maximum_learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)

def get_learning_rate(update_index):
    if update_index < warmup_steps:
        return maximum_learning_rate * (update_index + 1) / warmup_steps
    progress = (update_index - warmup_steps) / (training_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + cosine * (maximum_learning_rate - minimum_learning_rate)

@torch.no_grad()
def evaluate():
    model.eval()
    results = {}
    for split in ['train', 'validation']:
        total_loss = 0.0
        for _ in range(evaluation_batches):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            total_loss += loss.item()
        average_loss = total_loss / evaluation_batches
        perplexity = math.exp(min(average_loss, 20))
        results[split] = {'loss': average_loss, 'perplexity': perplexity}
    model.train()
    return results

def save_checkpoint(filename, completed_updates, best_validation_loss):
    checkpoint = {'completed_updates': completed_updates, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_validation_loss': best_validation_loss, 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(), 'config': {'vocab_size': vocab_size, 'context_length': context_length, 'batch_size': batch_size, 'd_model': d_model, 'number_of_layers': number_of_layers, 'number_of_heads': number_of_heads, 'head_size': head_size, 'mlp_hidden': mlp_hidden, 'training_steps': training_steps}}
    torch.save(checkpoint, filename)
completed_updates = 0
best_validation_loss = float('inf')
if RESUME and os.path.exists(LATEST_CHECKPOINT):
    print('\nLoading v0.2 latest checkpoint...')
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    completed_updates = checkpoint['completed_updates']
    best_validation_loss = checkpoint['best_validation_loss']
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    print('Completed updates:', f'{completed_updates:,}')
    print('Remaining updates:', f'{training_steps - completed_updates:,}')
if completed_updates == 0:
    print('\nINITIAL EVALUATION')
    initial_metrics = evaluate()
    initial_train_loss = initial_metrics['train']['loss']
    initial_val_loss = initial_metrics['validation']['loss']
    initial_ppl = initial_metrics['validation']['perplexity']
    best_validation_loss = initial_val_loss
    print(f'updates={0:5d} | train={initial_train_loss:.4f} | val={initial_val_loss:.4f} | ppl={initial_ppl:.2f}')
    save_checkpoint(BEST_CHECKPOINT, 0, best_validation_loss)
    save_checkpoint(LATEST_CHECKPOINT, 0, best_validation_loss)
print('\nTRAINING BLANK-0 v0.2\n')
model.train()
session_start = time.perf_counter()
report_start_time = time.perf_counter()
updates_since_report = 0
while completed_updates < training_steps:
    update_index = completed_updates
    learning_rate = get_learning_rate(update_index)
    for parameter_group in optimizer.param_groups:
        parameter_group['lr'] = learning_rate
    X, Y = get_batch('train')
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    completed_updates += 1
    updates_since_report += 1
    should_evaluate = completed_updates % evaluation_interval == 0
    if should_evaluate:
        torch.cuda.synchronize()
        now = time.perf_counter()
        tokens_processed = updates_since_report * tokens_per_update
        elapsed_since_report = now - report_start_time
        throughput = tokens_processed / elapsed_since_report
        metrics = evaluate()
        train_loss = metrics['train']['loss']
        validation_loss = metrics['validation']['loss']
        validation_perplexity = metrics['validation']['perplexity']
        current_lr = get_learning_rate(min(completed_updates, training_steps - 1))
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        total_predictions = completed_updates * tokens_per_update
        dataset_equivalents = total_predictions / train_tokens
        print(f'updates={completed_updates:5d} | train={train_loss:.4f} | val={validation_loss:.4f} | ppl={validation_perplexity:.2f} | lr={current_lr:.2e} | tok/s={throughput:,.0f} | data={dataset_equivalents:.3f}x | VRAM={reserved:.2f} GB')
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(BEST_CHECKPOINT, completed_updates, best_validation_loss)
            print('  -> new best checkpoint')
        save_checkpoint(LATEST_CHECKPOINT, completed_updates, best_validation_loss)
        report_start_time = time.perf_counter()
        updates_since_report = 0
torch.cuda.synchronize()
session_end = time.perf_counter()
final_metrics = evaluate()
final_train_loss = final_metrics['train']['loss']
final_validation_loss = final_metrics['validation']['loss']
final_perplexity = final_metrics['validation']['perplexity']
print('\nFINAL RESULTS')
print('Completed updates:', f'{completed_updates:,}')
print('Token predictions:', f'{completed_updates * tokens_per_update:,}')
print('Dataset equivalents:', round(completed_updates * tokens_per_update / train_tokens, 4))
print('Training loss:', round(final_train_loss, 4))
print('Validation loss:', round(final_validation_loss, 4))
print('Validation perplexity:', round(final_perplexity, 2))
if final_validation_loss < best_validation_loss:
    best_validation_loss = final_validation_loss
    save_checkpoint(BEST_CHECKPOINT, completed_updates, best_validation_loss)
    print('Final model is the new best checkpoint.')
print('Best validation loss:', round(best_validation_loss, 4))
save_checkpoint(FINAL_CHECKPOINT, completed_updates, best_validation_loss)
save_checkpoint(LATEST_CHECKPOINT, completed_updates, best_validation_loss)
print('\nFinal checkpoint saved:', FINAL_CHECKPOINT)
print('Latest checkpoint saved:', LATEST_CHECKPOINT)
print('Session time:', round(session_end - session_start, 2), 'seconds')
print('GPU allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 2), 'GB')
print('GPU reserved:', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 2), 'GB')
checkpoint = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print('\nLoaded best checkpoint.')
print('Best checkpoint updates:', checkpoint['completed_updates'])
print('Best checkpoint validation loss:', checkpoint['best_validation_loss'])

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
            top_values, _ = torch.topk(logits, top_k)
            threshold = top_values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return tokenizer.decode(generated[0].tolist())
prompts = ['Artificial intelligence is', 'The history of Europe', 'A computer network is', 'The human brain', 'In mathematics, a function']
for index, prompt in enumerate(prompts, start=1):
    print('\n========================================')
    print(f'GENERATED SAMPLE {index}')
    print('========================================\n')
    print(generate(prompt, number_of_tokens=250, temperature=0.8, top_k=50))
