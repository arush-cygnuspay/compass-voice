# Production Model Artifacts

This directory is the **canonical location** for all production NLU model bundles used by Compass Voice.

## Directory layout

```
artifacts/models/
├── intent/
│   └── distilbert-multihead-intent/   # HuggingFace multi-head intent classifier
│       ├── model.safetensors          ← Git LFS
│       ├── training_args.bin          ← Git LFS
│       ├── spm.model                  ← Git LFS
│       ├── config.json                ← normal Git
│       ├── tokenizer.json             ← normal Git
│       ├── tokenizer_config.json      ← normal Git
│       ├── added_tokens.json          ← normal Git
│       ├── special_tokens_map.json    ← normal Git
│       ├── labels_main.json           ← normal Git
│       ├── labels_sub.json            ← normal Git
│       ├── label_mappings.json        ← normal Git
│       └── best_model_summary.json    ← normal Git
└── slot/
    └── model-best/                    # spaCy NER slot model
        ├── ner/model                  ← Git LFS
        ├── transformer/model          ← Git LFS
        ├── vocab/vectors              ← Git LFS
        ├── vocab/key2row              ← Git LFS
        ├── vocab/lookups.bin          ← Git LFS
        ├── config.cfg                 ← normal Git
        ├── meta.json                  ← normal Git
        ├── ner/cfg                    ← normal Git
        ├── ner/moves                  ← normal Git
        ├── transformer/cfg            ← normal Git
        ├── vocab/strings.json         ← normal Git
        └── vocab/vectors.cfg          ← normal Git
```

## Runtime configuration

The runtime resolves model paths from the project root. Defaults:

| Env var | Default |
|---|---|
| `COMPASS_INTENT_MODEL_DIR` | `artifacts/models/intent/distilbert-multihead-intent` |
| `COMPASS_INTENT_LABELS_MAIN` | `artifacts/models/intent/distilbert-multihead-intent/labels_main.json` |
| `COMPASS_INTENT_LABELS_SUB` | `artifacts/models/intent/distilbert-multihead-intent/labels_sub.json` |
| `COMPASS_SLOT_MODEL_DIR` | `artifacts/models/slot/model-best` |

All paths support relative (resolved from project root) or absolute values.

## Replacing a model safely

1. Place the new model bundle in a staging directory, e.g. `artifacts/models/intent/distilbert-multihead-intent-v2/`.
2. Verify it loads: `COMPASS_INTENT_MODEL_DIR=artifacts/models/intent/distilbert-multihead-intent-v2 python -m app.bootstrap.runtime`
3. Once confirmed, swap the directory name (or update the env var in your deployment config).
4. Delete the old bundle only after the new one is validated.

## Git LFS — verification commands

```bash
# List all files tracked by LFS
git lfs ls-files

# Check attributes on a specific file
git check-attr -a -- artifacts/models/intent/distilbert-multihead-intent/model.safetensors
git check-attr -a -- artifacts/models/slot/model-best/ner/model

# After .gitattributes changes, re-register files with LFS:
git rm --cached -r artifacts/models/intent artifacts/models/slot
git add .gitattributes .gitignore artifacts/models/intent artifacts/models/slot
git lfs ls-files   # should list all binary blobs
git status
```

> **Do not commit without reviewing `git lfs ls-files` output first.**
> Binary blobs committed without LFS inflate the repo permanently.

## What stays in normal Git

Small, human-readable files that benefit from diffs:
- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `labels_main.json`, `labels_sub.json`, `label_mappings.json`
- `best_model_summary.json`, `meta.json`
- `config.cfg`, `ner/cfg`, `ner/moves`, `transformer/cfg`
- `vocab/strings.json`, `vocab/vectors.cfg`

## What is tracked by LFS

Large binary blobs that must not inflate Git history:
- `*.safetensors` — model weights
- `*.bin` — HuggingFace training args, spaCy vocab blobs
- `*.model` — SentencePiece tokenizer
- `model` (no extension) — spaCy NER + transformer component blobs
- `vectors`, `key2row` — spaCy vocab binary files
