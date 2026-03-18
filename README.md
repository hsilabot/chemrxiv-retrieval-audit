# chemrxiv-retrieval-audit

Reproducible scaffold to generate a ChemRxiv retrieval review packet (JSONL) using **full-corpus retrieval** and memory-safe chunking.

## What this does

- Loads `BASF-AI/ChemRxivRetrieval` (`corpus`, `queries`, `qrels/default`) from Hugging Face
- Retrieves top-k docs for each query using a SentenceTransformer model
- Collects target counts:
  - 25 successful examples (`gold doc in top-k`)
  - 25 unsuccessful examples (`gold doc not in top-k`)
- Writes one JSONL packet in MTEB-style shape with a `__meta__` header line

## Critical scope guarantee

**Batch/chunk sizes affect memory and speed only — not retrieval scope.**

This implementation searches every scanned query against **all corpus documents**. It does so memory-safely by:
1. encoding corpus in chunks,
2. scoring query-batch vs chunk,
3. keeping a running top-k per query across all chunks.

No corpus subsetting is performed.

---

## Setup (macOS Apple Silicon friendly)

### 1) Clone / enter repo

```bash
cd ~/workspace/chemrxiv-retrieval-audit
```

### 2) Create environment (uv recommended)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

(Alternative: `pip install -r requirements.txt`)

### 3) Optional HF auth for gated model

If using gated/private models (e.g. `BASF-AI/ChEmbed-prog`), set token locally.

Note: `generate_packet.py` hard-requires `trust_remote_code=True` for `BASF-AI/ChEmbed-prog`.

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=...
export $(grep -v '^#' .env | xargs)
```

---

## Generate packet

Default model is `BASF-AI/ChEmbed-prog`.

```bash
python scripts/generate_packet.py \
  --model-id BASF-AI/ChEmbed-prog \
  --success-target 25 \
  --failure-target 25 \
  --top-k 10 \
  --query-batch-size 8 \
  --corpus-chunk-size 1024 \
  --max-seq-length 512 \
  --device mps \
  --seed 42 \
  --output-path outputs/chemrxiv_packet_fullcorpus.jsonl

Notes:
- Split is treated as **strict** (no fallback). Default is `--split test`.
- The script forces `trust_remote_code=True` for `BASF-AI/ChEmbed-prog`.
```

### Suggested starting values for ~10GB headroom (M-series)

- `--device mps`
- `--query-batch-size 4` to `8`
- `--corpus-chunk-size 512` to `1024`
- `--max-seq-length 384` or `512`

If you see memory pressure, reduce `query-batch-size` first, then `corpus-chunk-size`.

---

## Validate packet

```bash
python scripts/check_packet.py \
  --path outputs/chemrxiv_packet_fullcorpus.jsonl \
  --expected-success 25 \
  --expected-failure 25 \
  --expected-top-k 10
```

---

## Output

- Packet: `outputs/*.jsonl`
- Schema docs: `docs/output_schema.md`

The JSONL format mirrors prior packet style:
- line 1 = `{"__meta__": ...}`
- remaining lines = examples with fields:
  `query_id, query_text, ground_truth_doc_id, ground_truth_score, ground_truth_text, top_k, retrieved, success`

Keys intentionally omitted (to match desired structure):
- `ground_truth_title`
- `retrieved[*].title`

---

## Notes

- This script does not commit or store tokens.
- It is intentionally conservative on memory usage by design.
- Runtime can be long because retrieval is full-corpus per scanned query.
