# ADR 001 — Mid-training batch-size downgrade semantics

**Status**: Accepted  
**Decided**: 2026-04-10 (implemented in commit 7c05c1d, PR 3)  
**Applies to**: `memory_guard.adapters.huggingface.MemoryGuardCallback`

---

## Context

When the background `RuntimeMonitor` thread signals that memory pressure has
crossed the critical threshold, it halves `current_batch_size`.  The question
is: when should `MemoryGuardCallback` apply that change to
`trainer.args.per_device_train_batch_size`?

Two options were considered.

### Option A — Immediate mutation (mid-epoch)

Apply the new batch size the moment `on_step_begin` fires:

```python
def on_step_begin(self, args, state, control, **kwargs):
    mon_bs = self._monitor.current_batch_size
    if mon_bs < args.per_device_train_batch_size:
        args.per_device_train_batch_size = mon_bs
        args.gradient_accumulation_steps *= ratio
```

**Advantage**: fastest response — the next step uses the smaller batch.

**Problems**:
- HuggingFace `Trainer` pre-builds its `DataLoader` with a fixed `batch_size`
  at the start of each epoch via `get_train_dataloader()`.  The dataloader is
  not rebuilt during the epoch, so mid-epoch mutation of `args` is silently
  ignored by the active dataloader; the actual batches keep arriving at the
  original size.
- Worse, if any downstream code reads `args.per_device_train_batch_size` during
  the epoch it will see an inconsistent value while the dataloader is still
  delivering the old batch size.
- Forcing a dataloader rebuild mid-epoch (`trainer.get_train_dataloader()`) is
  possible but invasive: it requires restarting the epoch iterator, skipping
  already-seen samples, and is not part of the public `TrainerCallback` contract.

### Option B — Deferred mutation at epoch boundary (chosen)

Record the pending downgrade in `on_step_begin`; apply it in `on_epoch_begin`.

```python
def on_step_begin(self, args, state, control, **kwargs):
    # Only record — never mutate args here
    if mon_bs < args.per_device_train_batch_size:
        self._pending_batch_size = min(self._pending_batch_size or inf, mon_bs)
        control.should_log = True   # flush warning to training log

def on_epoch_begin(self, args, state, control, **kwargs):
    # Apply at the safe point where HF rebuilds the DataLoader
    if self._pending_batch_size is not None:
        ratio = args.per_device_train_batch_size // self._pending_batch_size
        args.gradient_accumulation_steps *= ratio
        args.per_device_train_batch_size = self._pending_batch_size
        self._pending_batch_size = None
```

**Advantages**:
- Epoch boundaries are the natural safe point: HF calls `get_train_dataloader()`
  at the start of each epoch, so the new `per_device_train_batch_size` is picked
  up immediately by the rebuilt dataloader.
- `args` is consistent throughout the epoch — no mid-epoch invariant violations.
- Gradient accumulation is scaled by `old_bs // new_bs` atomically with the
  batch-size change, preserving the effective batch size:
  `per_device_train_batch_size × gradient_accumulation_steps = constant`.
- The pending state is tracked as the **minimum** observed value across all steps
  in the epoch, so multiple monitor drops before the boundary are collapsed into
  one clean change.

**Trade-off**: up to one full epoch of training at the higher batch size before
the downgrade takes effect.  Acceptable because:
1. The monitor fires warnings immediately via `control.should_log`, giving the
   user visibility before the change applies.
2. The epoch latency bound is bounded by `num_steps_per_epoch × step_time`,
   which is typically seconds to minutes — far shorter than the OOM window on
   Apple Silicon (where OOM is gradual, not instantaneous).
3. A single epoch at elevated batch size is unlikely to cause an OOM crash;
   the monitor threshold is set conservatively (critical at 85 %).

## Decision

**Defer to epoch boundary.**  `on_step_begin` records, `on_epoch_begin` applies.

## Consequences

- `MemoryGuardCallback` is safe to use with any HuggingFace `Trainer` or TRL
  `SFTTrainer` without patching or subclassing the data-loading machinery.
- If a user trains with `max_steps` and no full epochs (single-epoch or step-
  based training), the downgrade applies at the next `on_epoch_begin` even if
  the epoch contains more steps than `max_steps`.  In practice this means the
  downgrade may never apply in a single-epoch run — acceptable for v0.2;
  a future `on_step_end` flush option can be added if needed.
- The `_pending_batch_size` field must be cleared in both `on_epoch_begin`
  (normal path) and `on_train_end` (teardown), and reset in `on_train_begin`
  (re-use across Trainer instances).  This is implemented and tested.
