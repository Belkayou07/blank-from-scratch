import gc
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
DATA_FILE = 'blank_dataset_v2_tokens.bin'
METADATA_FILE = 'blank_dataset_v2_tokens_meta.json'
CONTEXT = 512
D_MODEL = 512
LAYERS = 10
HEADS = 8
HEAD_DIM = 64
MLP_HIDDEN = 1365
TARGET_TOKEN_PREDICTIONS = 285736960
BATCH_CANDIDATES = [16, 32, 48, 64, 80, 96, 112, 128]
OPT_WARMUP = 5
OPT_TIMED = 20
BATCH_WARMUP = 5
BATCH_TIMED = 20
COMPILE_WARMUP = 5
COMPILE_TIMED = 20
if not torch.cuda.is_available():
    raise RuntimeError('ROCm GPU unavailable.')
device = torch.device('cuda:0')
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__)
print('HIP:', torch.version.hip)
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2), 'GB')
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
VOCAB = metadata['vocab_size']
TOTAL_TOKENS = metadata['total_tokens']
TRAIN_TOKENS = metadata['train_tokens']
print('\nLoading dataset as INT32...')
disk_tokens = np.memmap(DATA_FILE, dtype=np.uint16, mode='r')
if len(disk_tokens) != TOTAL_TOKENS:
    raise RuntimeError('Dataset size mismatch.')
cpu_tokens = torch.from_numpy(np.asarray(disk_tokens, dtype=np.int32))
all_tokens = cpu_tokens.to(device)
del cpu_tokens
del disk_tokens
train_data = all_tokens[:TRAIN_TOKENS]
dataset_memory = all_tokens.numel() * all_tokens.element_size() / 1024 ** 3
print('Dataset GPU memory:', round(dataset_memory, 2), 'GB')
offsets = torch.arange(CONTEXT, device=device)

def get_batch(batch_size):
    starts = torch.randint(0, TRAIN_TOKENS - CONTEXT - 1, (batch_size,), device=device)
    positions = starts[:, None] + offsets[None, :]
    X = train_data[positions].long()
    Y = train_data[positions + 1].long()
    return (X, Y)

class RMSNorm(nn.Module):

    def __init__(self, dimension, epsilon=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, x):
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.epsilon) * self.weight

def apply_rotary(x, cosine, sine):
    even = x[..., 0::2]
    odd = x[..., 1::2]
    new_even = even * cosine - odd * sine
    new_odd = even * sine + odd * cosine
    return torch.stack([new_even, new_odd], dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):

    def __init__(self, dimension, maximum_length, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dimension, 2).float() / dimension)
        positions = torch.arange(maximum_length).float()
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer('cosine', frequencies.cos(), persistent=False)
        self.register_buffer('sine', frequencies.sin(), persistent=False)

    def forward(self, Q, K):
        T = Q.shape[-2]
        cosine = self.cosine[:T].to(dtype=Q.dtype)[None, None, :, :]
        sine = self.sine[:T].to(dtype=Q.dtype)[None, None, :, :]
        return (apply_rotary(Q, cosine, sine), apply_rotary(K, cosine, sine))

class Attention(nn.Module):

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.key = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.value = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.output = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.rotary = RotaryEmbedding(HEAD_DIM, CONTEXT)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.reshape(B, T, HEADS, HEAD_DIM).transpose(1, 2)
        K = K.reshape(B, T, HEADS, HEAD_DIM).transpose(1, 2)
        V = V.reshape(B, T, HEADS, HEAD_DIM).transpose(1, 2)
        Q, K = self.rotary(Q, K)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D_MODEL)
        return self.output(out)

class SwiGLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, MLP_HIDDEN, bias=False)
        self.up = nn.Linear(D_MODEL, MLP_HIDDEN, bias=False)
        self.down = nn.Linear(MLP_HIDDEN, D_MODEL, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.attention = Attention()
        self.mlp_norm = RMSNorm(D_MODEL)
        self.mlp = SwiGLU()

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class BlankOne(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.final_norm = RMSNorm(D_MODEL)

    def forward(self, X, targets=None):
        x = self.token_embedding(X)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        return (logits, loss)

def bf16_context():
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def create_optimizer(model, mode):
    kwargs = dict(params=model.parameters(), lr=0.0001, betas=(0.9, 0.95), weight_decay=0.1)
    if mode == 'fused':
        return torch.optim.AdamW(**kwargs, fused=True)
    elif mode == 'foreach':
        return torch.optim.AdamW(**kwargs, foreach=True)
    elif mode == 'default':
        return torch.optim.AdamW(**kwargs)
    else:
        raise ValueError(mode)

def train_update(model, optimizer, batch_size):
    X, Y = get_batch(batch_size)
    optimizer.zero_grad(set_to_none=True)
    with bf16_context():
        _, loss = model(X, Y)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss.')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss

def benchmark(model, optimizer, batch_size, warmup, timed):
    model.train()
    for _ in range(warmup):
        train_update(model, optimizer, batch_size)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    last_loss = None
    for _ in range(timed):
        last_loss = train_update(model, optimizer, batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_tokens = timed * batch_size * CONTEXT
    throughput = total_tokens / elapsed
    peak_allocated = torch.cuda.max_memory_allocated(0) / 1024 ** 3
    peak_reserved = torch.cuda.max_memory_reserved(0) / 1024 ** 3
    return {'tok_s': throughput, 'elapsed': elapsed, 'loss': last_loss.item(), 'allocated': peak_allocated, 'reserved': peak_reserved}
temporary_model = BlankOne().to(device)
parameter_count = sum((p.numel() for p in temporary_model.parameters()))
print('\nParameters:', f'{parameter_count:,}')
if parameter_count != 33560064:
    raise RuntimeError('Blank-1 parameter count mismatch.')
del temporary_model
cleanup()
print('\n' + '=' * 72)
print('PHASE 1 — OPTIMIZER BENCHMARK')
print('=' * 72)
optimizer_results = []
for optimizer_mode in ['default', 'foreach', 'fused']:
    cleanup()
    print(f'\nTesting AdamW: {optimizer_mode}')
    try:
        torch.manual_seed(123)
        model = BlankOne().to(device)
        optimizer = create_optimizer(model, optimizer_mode)
        result = benchmark(model, optimizer, batch_size=16, warmup=OPT_WARMUP, timed=OPT_TIMED)
        result['optimizer'] = optimizer_mode
        optimizer_results.append(result)
        print('Tokens/sec:', f"{result['tok_s']:,.0f}")
        print('Peak reserved:', f"{result['reserved']:.2f} GB")
    except Exception as error:
        print('FAILED:', type(error).__name__, error)
    finally:
        try:
            del optimizer
        except:
            pass
        try:
            del model
        except:
            pass
        cleanup()
if not optimizer_results:
    raise RuntimeError('Every optimizer benchmark failed.')
best_optimizer = max(optimizer_results, key=lambda x: x['tok_s'])
BEST_OPTIMIZER = best_optimizer['optimizer']
print('\nFASTEST OPTIMIZER:', BEST_OPTIMIZER)
print('Speed:', f"{best_optimizer['tok_s']:,.0f}", 'tok/s')
print('\n' + '=' * 72)
print('PHASE 2 — BATCH SIZE AUTOTUNE')
print('=' * 72)
batch_results = []
for batch_size in BATCH_CANDIDATES:
    cleanup()
    print(f'\nTesting batch {batch_size}...')
    try:
        torch.manual_seed(123)
        model = BlankOne().to(device)
        optimizer = create_optimizer(model, BEST_OPTIMIZER)
        result = benchmark(model, optimizer, batch_size=batch_size, warmup=BATCH_WARMUP, timed=BATCH_TIMED)
        result['batch'] = batch_size
        batch_results.append(result)
        print('Tokens/update:', f'{batch_size * CONTEXT:,}')
        print('Tokens/sec:', f"{result['tok_s']:,.0f}")
        print('ms/update:', round(result['elapsed'] / BATCH_TIMED * 1000, 2))
        print('Peak allocated:', f"{result['allocated']:.2f} GB")
        print('Peak reserved:', f"{result['reserved']:.2f} GB")
    except torch.cuda.OutOfMemoryError:
        print('OUT OF MEMORY.')
        cleanup()
        break
    except Exception as error:
        print('FAILED:', type(error).__name__, error)
    finally:
        try:
            del optimizer
        except:
            pass
        try:
            del model
        except:
            pass
        cleanup()
if not batch_results:
    raise RuntimeError('Every batch benchmark failed.')
maximum_batch_speed = max((x['tok_s'] for x in batch_results))
eligible_batches = [result for result in batch_results if result['tok_s'] >= 0.97 * maximum_batch_speed]
best_batch_result = min(eligible_batches, key=lambda x: x['batch'])
BEST_BATCH = best_batch_result['batch']
print('\nMAXIMUM OBSERVED BATCH SPEED:', f'{maximum_batch_speed:,.0f}', 'tok/s')
print('SELECTED BATCH:', BEST_BATCH)
print('Selected speed:', f"{best_batch_result['tok_s']:,.0f}", 'tok/s')
print('\n' + '=' * 72)
print('PHASE 3 — TORCH.COMPILE')
print('=' * 72)
engine_results = []
print('\nTesting eager...')
cleanup()
torch.manual_seed(123)
raw_model = BlankOne().to(device)
optimizer = create_optimizer(raw_model, BEST_OPTIMIZER)
eager_result = benchmark(raw_model, optimizer, batch_size=BEST_BATCH, warmup=COMPILE_WARMUP, timed=COMPILE_TIMED)
eager_result['engine'] = 'eager'
engine_results.append(eager_result)
print('Eager:', f"{eager_result['tok_s']:,.0f}", 'tok/s')
del optimizer
del raw_model
cleanup()
for compile_mode in ['reduce-overhead', 'max-autotune']:
    print(f"\nTesting torch.compile(mode='{compile_mode}')...")
    cleanup()
    try:
        torch.manual_seed(123)
        raw_model = BlankOne().to(device)
        compiled_model = torch.compile(raw_model, mode=compile_mode, fullgraph=False)
        optimizer = create_optimizer(raw_model, BEST_OPTIMIZER)
        result = benchmark(compiled_model, optimizer, batch_size=BEST_BATCH, warmup=COMPILE_WARMUP, timed=COMPILE_TIMED)
        result['engine'] = 'compile:' + compile_mode
        engine_results.append(result)
        print('Tokens/sec:', f"{result['tok_s']:,.0f}")
        print('Peak reserved:', f"{result['reserved']:.2f} GB")
    except Exception as error:
        print('COMPILE FAILED:')
        print(type(error).__name__, ':', error)
    finally:
        try:
            del optimizer
        except:
            pass
        try:
            del compiled_model
        except:
            pass
        try:
            del raw_model
        except:
            pass
        cleanup()
best_engine = max(engine_results, key=lambda x: x['tok_s'])
BEST_ENGINE = best_engine['engine']
BEST_SPEED = best_engine['tok_s']
tokens_per_update = BEST_BATCH * CONTEXT
required_updates = math.ceil(TARGET_TOKEN_PREDICTIONS / tokens_per_update)
actual_predictions = required_updates * tokens_per_update
raw_seconds = actual_predictions / BEST_SPEED
raw_minutes = raw_seconds / 60
estimated_total_minutes = raw_minutes * 1.08
print('\n' + '=' * 90)
print('AUTOTUNE RESULTS')
print('=' * 90)
print('\nOPTIMIZERS')
for result in optimizer_results:
    print(f"{result['optimizer']:10s}  {result['tok_s']:>10,.0f} tok/s")
print('\nBATCH SIZES')
for result in batch_results:
    print(f"batch={result['batch']:3d}  {result['tok_s']:>10,.0f} tok/s  VRAM={result['reserved']:.2f} GB")
print('\nEXECUTION ENGINES')
for result in engine_results:
    print(f"{result['engine']:28s}  {result['tok_s']:>10,.0f} tok/s  VRAM={result['reserved']:.2f} GB")
print('\n' + '=' * 90)
print('RECOMMENDED FULL TRAINING CONFIGURATION')
print('=' * 90)
print('Optimizer:', BEST_OPTIMIZER)
print('Batch size:', BEST_BATCH)
print('Context:', CONTEXT)
print('Execution engine:', BEST_ENGINE)
print('Precision: BF16')
print('Attention: SDPA')
print('Measured speed:', f'{BEST_SPEED:,.0f}', 'tokens/sec')
print('\nFULL TOKEN BUDGET')
print('Target predictions:', f'{TARGET_TOKEN_PREDICTIONS:,}')
print('Tokens/update:', f'{tokens_per_update:,}')
print('Required updates:', f'{required_updates:,}')
print('Actual predictions:', f'{actual_predictions:,}')
print('\nESTIMATED FULL-RUN TIME')
print('Raw training estimate:', round(raw_minutes, 2), 'minutes')
print('Including ~8% overhead:', round(estimated_total_minutes, 2), 'minutes')
print('\nDONE')
