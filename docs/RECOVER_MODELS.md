# Recovering downloaded model weights

Weights are cached outside the repository, in the Hugging Face cache (`~/.cache/huggingface/hub`). They are not committed. This note records what was
deleted to free disk, and the exact commands to bring each model back.

| Model | Cache directory | Size | Needed for | Status |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | `models--Qwen--Qwen3-0.6B` | 1.4 GB | everything in the main study, all organisms, the word game | **kept** |
| `Qwen/Qwen2.5-1.5B-Instruct` | `models--Qwen--Qwen2.5-1.5B-Instruct` (snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) | 2.9 GB | second-model steering runs, label-shuffle on 1.5B | **deleted 2026-09-30** |

The 1.5B runs are already finished and committed (`results/qwen25_1p5b/`). Re-download only to re-run them or to run the 1.5B label-shuffle control.

## Recover Qwen2.5-1.5B-Instruct (about 2.9 GB, needs about 6 GB free while downloading)

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))"
```

Then the 1.5B jobs (bf16 is required on an 8 GB machine; fp32 pushes it into swap):

```bash
LOYALTY_DTYPE=bfloat16 LOYALTY_MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 src/real_model.py --out results/qwen25_1p5b/real_model.json
LOYALTY_DTYPE=bfloat16 LOYALTY_MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 src/label_shuffle_control.py \
  --principals China Russia USA Uruguay --out results/qwen25_1p5b/label_shuffle.json
```

## Recover Qwen3-0.6B if it is ever removed

```bash
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))"
```

## Free-disk checklist (8 GB machine)

* Swap grows with memory pressure and shows up as lost disk; close other apps before long runs and stop background jobs if free space drops below about 3 GB.
* Organism adapters are about 20 MB each and are ignored by git; seed-repeat adapters are deleted after their audit.
* `pip`/`huggingface_hub` caches can be inspected with `du -sh ~/.cache/huggingface ~/Library/Caches/pip`.
