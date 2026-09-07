import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ==============================================================================
# CONFIGURATION
# ==============================================================================
FILE_PATH = "input.txt"
EPOCHS = 10
BATCH_SIZE = 8
SEQ_LEN = 512        # Context length of the transformer
N_HORIZON = 6       # (N) How many future tokens the concept head should average
LAMBDA_C = 2.0      # Weight of the concept loss vs next-token loss
EVAL_INTERVAL = 100 # Evaluate val loss every X steps
GEN_INTERVAL = 1000 # Generate sample text every X steps

# --- Concept-mechanism upgrades ---
USE_EMA_TARGETS = True      # Build targets from an EMA copy of token_emb (BYOL/data2vec-style; kills the moving-target loop)
EMA_DECAY = 0.999
CONCEPT_LOSS_TYPE = "mse"   # "mse" = cosine-like regression | "infonce" = CPC-style contrastive over in-batch negatives
INFONCE_TEMP = 0.1          # Temperature for the infonce variant
CONCEPT_CENTER_TARGETS = True  # Subtract the batch-mean (corpus-centroid) direction from targets before normalizing
                               # (data2vec-style centering). Without it, a CONSTANT prediction already sits ~4% from the
                               # floor (observed in run logs) and the head learns nothing position-specific.

# --- Optimization / checkpointing ---
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.1          # Applied to 2D non-embedding weights only
CLIP_GRAD_NORM = 1.0
WARMUP_FRAC = 0.05          # Fraction of steps for LR warmup, then cosine decay
SEED = 42
CHECKPOINT_PATH = "concept_transformer_ckpt.pt"  # Delete this file to retrain from scratch

# Model params (Small)
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
MAX_NEW_TOKENS = 50 # How many tokens to generate during the generation phase
CHAT_MAX_TOKENS = 150 # How many tokens to generate per chat turn

torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# ==============================================================================
# 1. DATA SETUP & TIKTOKEN
# ==============================================================================
# Create a dummy input.txt if it doesn't exist so the script runs out of the box
if not os.path.exists(FILE_PATH):
    print(f"{FILE_PATH} not found. Creating a dummy dataset...")
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 1000
    with open(FILE_PATH, "w") as f:
        f.write(dummy_text)

# Tokenize with tiktoken (GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text_data = f.read()

print("Tokenizing data...")
tokens = enc.encode(text_data)
data = torch.tensor(tokens, dtype=torch.long)

# Split into Train (90%) and Val (10%)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]
print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}, Vocab: {vocab_size:,}")

def get_batch(split):
    # We need sequences of length (SEQ_LEN + N_HORIZON + 1) to get both
    # the next-token targets and the N-horizon future embeddings.
    d = train_data if split == 'train' else val_data
    max_idx = max(1, len(d) - SEQ_LEN - N_HORIZON - 1)
    ix = torch.randint(0, max_idx, (BATCH_SIZE,))
    
    # x_full contains the context + the future horizon needed for the targets
    x_full = torch.stack([d[i : i + SEQ_LEN + N_HORIZON] for i in ix])
    
    # y contains the standard next-token targets
    y = torch.stack([d[i + 1 : i + SEQ_LEN + 1] for i in ix])
    
    return x_full.to(device), y.to(device)

# ==============================================================================
# 2. MODEL ARCHITECTURE
# ==============================================================================
class ConceptTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        
        # Standard causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=4*D_MODEL, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        
        # Standard language modeling head
        self.lm_head = nn.Linear(D_MODEL, vocab_size)
        
        # THE MECHANISM: Auxiliary head to predict future conceptual centroid
        self.concept_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL)
        )
        
        # EMA "teacher" embedding table (BYOL/data2vec-style): concept targets are
        # read from this slowly-updating copy instead of the live token_emb, so the
        # target geometry cannot be chased around by training itself.
        if USE_EMA_TARGETS:
            self.target_emb = nn.Embedding(vocab_size, D_MODEL)
            with torch.no_grad():
                self.target_emb.weight.copy_(self.token_emb.weight)
            self.target_emb.weight.requires_grad_(False)

    @torch.no_grad()
    def update_target_emb(self):
        """Momentum update of the EMA target table; call after optimizer.step()."""
        if USE_EMA_TARGETS:
            self.target_emb.weight.mul_(EMA_DECAY).add_(
                self.token_emb.weight.detach(), alpha=1.0 - EMA_DECAY)

    def concept_target_table(self):
        """Which embedding table to build concept targets from."""
        return self.target_emb if USE_EMA_TARGETS else self.token_emb

    def forward(self, input_ids):
        B, T = input_ids.size()
        
        # 1. Embeddings
        tok_emb = self.token_emb(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = tok_emb + self.pos_emb(pos)
        
        # 2. Causal Mask (float -inf, unambiguous across torch versions; bool-mask
        #    polarity for TransformerEncoder flipped between versions)
        mask = torch.full((T, T), float('-inf'), device=device).triu(diagonal=1)
        
        # 3. Transformer Forward (explicit mask alone; mask+is_causal together breaks fast paths)
        hidden = self.transformer(x, mask=mask)
        
        # 4. Heads
        logits = self.lm_head(hidden)
        concept_preds = self.concept_head(hidden)
        
        return logits, concept_preds

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        was_training = self.training
        self.eval()
        idx = start_tokens
        try:
            for _ in range(max_new_tokens):
                # Crop context to max SEQ_LEN
                idx_cond = idx[:, -SEQ_LEN:]
                logits, _ = self(idx_cond)
                
                # Get last token logits (temperature-scaled)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)  # restore prior mode instead of forcing train()
        return idx

# ==============================================================================
# 3. TRAINING LOGIC & LOSS CALCULATION
# ==============================================================================
def build_concept_targets(emb_table, x_full):
    """Vectorized sliding-window mean of the next N_HORIZON token embeddings.
    Slicing [:, 1:] shifts by one, so window i covers tokens i+1 .. i+N_HORIZON.
    (Replaces the old 512-iteration Python loop.)"""
    full_emb = emb_table(x_full)  # [B, SEQ_LEN + N_HORIZON, D_MODEL]
    centroids = full_emb[:, 1:, :].unfold(1, N_HORIZON, 1).mean(dim=-1)  # [B, SEQ_LEN, D_MODEL]
    if CONCEPT_CENTER_TARGETS:
        # Drop the dominant shared direction (frequent-token / corpus centroid):
        # without this, predicting a CONSTANT vector nearly reaches the floor and
        # the head goes vacuous. Centering forces position-specific prediction.
        centroids = centroids - centroids.mean(dim=(0, 1), keepdim=True)
    return centroids

def concept_loss_fn(concept_preds, concept_targets):
    if CONCEPT_LOSS_TYPE == "infonce":
        # CPC-style contrastive: rank the true future centroid above all other
        # in-batch centroids (defeats the mean-collapse of regression targets).
        p = F.normalize(concept_preds.reshape(-1, D_MODEL), dim=-1)
        t = F.normalize(concept_targets.reshape(-1, D_MODEL), dim=-1)
        logits = (p @ t.T) / INFONCE_TEMP
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)
    # Default: MSE on normalized vectors (~ cosine distance), scale-independent
    pred_norm = F.normalize(concept_preds, dim=-1)
    targ_norm = F.normalize(concept_targets, dim=-1)
    return F.mse_loss(pred_norm, targ_norm)

@torch.no_grad()
def concept_floor(concept_targets):
    """Baseline the concept head must beat.
    MSE: loss of the best constant predictor (the mean target direction).
    InfoNCE: chance level, i.e. uniform log-probability over the candidate pool.
    Skill = 1 - concept_loss/floor: 0% = no better than constant, 100% = perfect."""
    if CONCEPT_LOSS_TYPE == "infonce":
        n = concept_targets.reshape(-1, D_MODEL).shape[0]
        return torch.tensor(math.log(n))
    targ_norm = F.normalize(concept_targets, dim=-1)
    mean_dir = F.normalize(targ_norm.reshape(-1, D_MODEL).mean(dim=0), dim=0)
    return F.mse_loss(mean_dir.expand_as(targ_norm), targ_norm)

# ---- Checkpoint resume: if one exists we skip training and enter chat mode ----
ckpt = None
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"Found checkpoint '{CHECKPOINT_PATH}' - skipping training, entering chat mode.")
    globals().update(ckpt["config"])  # restore the architecture the weights were trained with

model = ConceptTransformer().to(device)

def save_checkpoint(path, step):
    torch.save({
        "model": model.state_dict(),
        "config": {
            "D_MODEL": D_MODEL, "N_HEAD": N_HEAD, "N_LAYERS": N_LAYERS,
            "SEQ_LEN": SEQ_LEN, "N_HORIZON": N_HORIZON,
            "LAMBDA_C": LAMBDA_C, "CONCEPT_LOSS_TYPE": CONCEPT_LOSS_TYPE,
            "USE_EMA_TARGETS": USE_EMA_TARGETS,
        },
        "step": step,
    }, path)
    print(f"Checkpoint saved to {path}")

EVAL_CACHE = {}
def eval_batches(split):
    """Fixed, seeded eval batches (created once, then reused). Every eval now
    scores the SAME windows, so step-to-step changes reflect the model, not
    batch resampling (which previously moved loss and floor in lockstep)."""
    if split not in EVAL_CACHE:
        g = torch.Generator()
        g.manual_seed(SEED)
        d = train_data if split == 'train' else val_data
        max_idx = max(1, len(d) - SEQ_LEN - N_HORIZON - 1)
        rows = torch.randint(0, max_idx, (10, BATCH_SIZE), generator=g).tolist()
        EVAL_CACHE[split] = [
            (
                torch.stack([d[i : i + SEQ_LEN + N_HORIZON] for i in row]).to(device),
                torch.stack([d[i + 1 : i + SEQ_LEN + 1] for i in row]).to(device),
            )
            for row in rows
        ]
    return EVAL_CACHE[split]

@torch.no_grad()
def estimate_loss():
    """CE and concept loss reported SEPARATELY: the overfitting claim must be
    measurable on the LM objective alone (the old joint metric conflated them)."""
    was_training = model.training
    model.eval()
    out = {}
    for split in ['train', 'val']:
        ce_losses = torch.zeros(10)
        concept_losses = torch.zeros(10)
        floors = torch.zeros(10)
        for k, (x_full, y) in enumerate(eval_batches(split)):
            x = x_full[:, :SEQ_LEN]

            logits, concept_preds = model(x)
            ce_losses[k] = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)).item()

            concept_targets = build_concept_targets(model.concept_target_table(), x_full)
            concept_losses[k] = concept_loss_fn(concept_preds, concept_targets).item()
            floors[k] = concept_floor(concept_targets).item()
        out[split + '_ce'] = ce_losses.mean().item()
        out[split + '_concept'] = concept_losses.mean().item()
        out[split + '_floor'] = floors.mean().item()
    model.train(was_training)
    return out

def chat_mode():
    print("\n" + "=" * 70)
    print("CHAT MODE - type a prompt and the model continues it.")
    print("Commands: /temp <float>  (sampling temperature)  |  /exit")
    print("=" * 70)
    temperature = 0.8
    model.eval()
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat ended]")
            break
        if not user_text:
            continue
        if user_text.lower() in ("/exit", "exit", "quit", "q"):
            break
        if user_text.startswith("/temp"):
            try:
                temperature = float(user_text.split()[1])
                print(f"(temperature = {temperature})")
            except (IndexError, ValueError):
                print("usage: /temp 0.8")
            continue
        start_tokens = torch.tensor(enc.encode(user_text), dtype=torch.long, device=device).unsqueeze(0)
        if start_tokens.numel() == 0:
            print("(could not tokenize input)")
            continue
        out_tokens = model.generate(start_tokens, max_new_tokens=CHAT_MAX_TOKENS, temperature=temperature)
        print("Model:", enc.decode(out_tokens[0].tolist()))

if ckpt is not None:
    # ---- Chat mode: trained checkpoint found ----
    model.load_state_dict(ckpt["model"])
    print(f"Loaded weights ({ckpt.get('step', '?')} steps trained). Delete '{CHECKPOINT_PATH}' to retrain.")
    chat_mode()
else:
    # ---- Training mode ----
    # Weight decay only on 2D non-embedding weights (embeddings/biases/norms excluded)
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen EMA target table
        (no_decay_params if (p.dim() < 2 or "emb" in name) else decay_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=LEARNING_RATE, fused=(device == 'cuda')
    )

    # LR schedule: linear warmup then cosine decay
    steps_per_epoch = len(train_data) // (BATCH_SIZE * SEQ_LEN)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Starting Training: {EPOCHS} Epochs | {total_steps} Total Steps")
    model.train()

    for step in range(total_steps):
        x_full, y = get_batch('train')
        x = x_full[:, :SEQ_LEN]  # The autoregressive input

        # 1. Forward pass
        logits, concept_preds = model(x)

        # 2. Standard Next-Token Loss
        ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        # 3. Concept Loss (The Mechanism)
        # Targets come from the EMA teacher table (or the live table, detached),
        # so no gradient flows through the target either way.
        with torch.no_grad():
            concept_targets = build_concept_targets(model.concept_target_table(), x_full)
        concept_loss = concept_loss_fn(concept_preds, concept_targets)

        # 4. Total loss with lambda warmup: CE organizes the embedding geometry
        #    first, then the concept signal ramps in. Lambda ramps over the SAME
        #    window as the LR warmup, so it never reaches full weight while the
        #    LR is still climbing (matters on long runs like this one).
        lambda_c_t = LAMBDA_C * min(1.0, (step + 1) / warmup_steps)
        loss = ce_loss + lambda_c_t * concept_loss

        # 5. Backprop (with grad clipping) + EMA teacher update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        model.update_target_emb()

        # ================= EVALUATION =================
        if step % EVAL_INTERVAL == 0 or step == total_steps - 1:
            losses = estimate_loss()
            val_ppl = math.exp(min(losses['val_ce'], 20.0))
            floor = losses['val_floor']
            skill = 1.0 - losses['val_concept'] / floor if floor > 0 else float('nan')
            print(f"Step {step}/{total_steps} | lr {scheduler.get_last_lr()[0]:.2e} | lambda_c {lambda_c_t:.2f}")
            print(f"  CE:      train {losses['train_ce']:.4f} | val {losses['val_ce']:.4f} (ppl {val_ppl:.1f})")
            if CONCEPT_LOSS_TYPE == "mse":
                # MSE on unit vectors: cos = 1 - (D/2)*MSE, so report readable cosines
                cos_model = 1.0 - (D_MODEL / 2.0) * losses['val_concept']
                cos_const = 1.0 - (D_MODEL / 2.0) * floor
                print(f"  Concept: val {losses['val_concept']:.5f} | floor {floor:.5f} | skill {100*skill:.1f}% "
                      f"(cos {cos_model:.3f} vs const {cos_const:.3f})")
                if lambda_c_t >= 0.99 * LAMBDA_C and skill < 0.10:
                    print("  WARNING: skill < 10% - the head barely beats a constant predictor")
            else:
                print(f"  Concept: val {losses['val_concept']:.4f} | chance {floor:.4f}")
                if lambda_c_t >= 0.99 * LAMBDA_C and losses['val_concept'] > 0.75 * floor:
                    print("  WARNING: infonce loss near chance level - head not discriminating")

        # ================= GENERATION =================
        if step % GEN_INTERVAL == 0 and step > 0:
            print("\n--- Generating Text (Iteration {}) ---".format(step))
            start_text = "The "
            start_tokens = torch.tensor(enc.encode(start_text), dtype=torch.long, device=device).unsqueeze(0)

            generated_tokens = model.generate(start_tokens, max_new_tokens=MAX_NEW_TOKENS)
            generated_text = enc.decode(generated_tokens[0].tolist())

            print(f"Output:\n{generated_text}")
            print("--------------------------------------\n")

    # ---- End of training: save checkpoint, then chat ----
    save_checkpoint(CHECKPOINT_PATH, total_steps)
    print("Training Complete.")
    chat_mode()