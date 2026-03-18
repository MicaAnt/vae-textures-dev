# Fine-Tuning Experiments from a Pretrained VAE (`model_param`)

This note is a **first practical approach** to organize your fine-tuning week using the pretrained checkpoint in `model_param`.

---

## Short answers to your 2 questions

### 1) Can I start from `train_smoke_cpu_002.py`?

**Yes.** It is a good starting scaffold because it already:

- builds the same model family (`DisentangleVAE`) with explicit submodules,
- runs train/validation loops,
- logs key losses,
- saves checkpoints.

But for real fine-tuning, convert it from a smoke script into a configurable experiment runner.

### 2) What are the options to freeze parts of the pretrained model?

Given this architecture, your main blocks are:

- `chd_encoder` (chord encoder)
- `rhy_encoder` (texture/rhythm encoder)
- `decoder` (piano tree decoder)
- `chd_decoder` (chord decoder)

So your freezing possibilities include:

1. **No freeze (full fine-tune)**
2. **Freeze both encoders; train only decoders**
3. **Freeze decoders; train only encoders**
4. **Freeze only `chd_encoder`**
5. **Freeze only `rhy_encoder`**
6. **Freeze only `chd_decoder`**
7. **Freeze only `decoder`**
8. **Gradual unfreezing** (start frozen, unfreeze stage by stage)
9. **Partial freeze by layer name** inside each block (advanced step)

A strong first week usually tests (1), (2), (4), (5), and (8).

---

## Week mindset: optimize learning signal, not just loss

When fine-tuning VAEs, avoid “just run and see.” Use a controlled mindset:

1. **Change one variable at a time** (freeze policy first, then LR, then beta/teacher forcing).
2. **Keep one stable baseline** (same seed, split, batch size) to compare runs.
3. **Track decomposition of losses** (`recon`, `kl`, `chord`) to detect collapse or drift.
4. **Prefer shorter, high-quality experiments** over many noisy runs.

---

## First experiment plan (practical and consolidated)

## Step 0 — Reproducible baseline

- Load pretrained checkpoint from `model_param`.
- Train for a small fixed budget (for example 5k–20k steps).
- Log train/val of: total loss, recon, KL, chord loss.
- Save “best val” and “last” checkpoints.

This baseline is your comparison anchor.

## Step 1 — Implement clean checkpoint loading

Use strict loading first; if key mismatch appears, switch to `strict=False` and log missing/unexpected keys.

```python
state = torch.load(pretrained_path, map_location=device)
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)
```

## Step 2 — Implement freeze policies as a single switch

Create a CLI argument like `--freeze-policy` with values:

- `none`
- `encoders`
- `decoders`
- `chd_encoder`
- `rhy_encoder`
- `chd_decoder`
- `decoder`

Then apply:

```python
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def apply_freeze_policy(model, policy: str):
    # unfreeze all first
    for p in model.parameters():
        p.requires_grad = True

    if policy == "none":
        return
    if policy == "encoders":
        set_requires_grad(model.chd_encoder, False)
        set_requires_grad(model.rhy_encoder, False)
    elif policy == "decoders":
        set_requires_grad(model.decoder, False)
        set_requires_grad(model.chd_decoder, False)
    elif policy == "chd_encoder":
        set_requires_grad(model.chd_encoder, False)
    elif policy == "rhy_encoder":
        set_requires_grad(model.rhy_encoder, False)
    elif policy == "decoder":
        set_requires_grad(model.decoder, False)
    elif policy == "chd_decoder":
        set_requires_grad(model.chd_decoder, False)
    else:
        raise ValueError(f"unknown policy: {policy}")
```

Build optimizer only with trainable params:

```python
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
)
```

## Step 3 — Define a compact experiment matrix

Start with 4 runs:

- **E0**: `freeze=none`, `lr=1e-4` (baseline fine-tune)
- **E1**: `freeze=encoders`, `lr=3e-4`
- **E2**: `freeze=chd_encoder`, `lr=2e-4`
- **E3**: `freeze=rhy_encoder`, `lr=2e-4`

Keep everything else identical.

## Step 4 — Add gradual unfreezing (second wave)

If E1 works reasonably:

- phase A: freeze encoders for first 30% of steps,
- phase B: unfreeze `rhy_encoder`,
- phase C: unfreeze all.

Use lower LR after each unfreeze (for example multiply by 0.5).

## Step 5 — Evaluate like a researcher

For each run, record:

- best val total loss
- best val recon
- best val KL
- best val chord loss
- sample quality notes (musical coherence, rhythmic consistency)

Pick next iteration from evidence, not intuition only.

---

## Practical best practices (important for VAE fine-tuning)

1. **Use a lower LR than from-scratch training** (often 2x to 10x lower).
2. **Watch KL behavior carefully**:
   - KL too close to zero for long time may indicate posterior collapse.
3. **Prefer early stopping on validation** for transfer settings.
4. **Keep seed fixed for comparison**, then rerun top candidates with 2–3 seeds.
5. **Separate “stability runs” from “quality runs”**:
   - first runs are short to detect direction,
   - later runs are longer only for best policies.
6. **Never compare experiments with different data splits** in the same conclusion.

---

## Suggested minimum TODOs before Monday

1. Add args:
   - `--pretrained-path`
   - `--freeze-policy`
   - `--run-name`
2. Add function `apply_freeze_policy`.
3. Add optimizer filtering by `requires_grad`.
4. Add logging folder per run name.
5. Run E0–E3 with same step budget.
6. Build a single markdown report with a result table.

---

## Final note

Your script is already a good skeleton. The key upgrade is to turn it into a **controlled experiment driver**. If you do that first, your whole week becomes simpler: each run answers one clear hypothesis.

