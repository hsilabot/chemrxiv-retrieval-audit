# Output schema (`packet.jsonl`)

The file is JSONL with:
1. First line: `{"__meta__": {...}}`
2. Subsequent lines: one retrieval example per line

## Meta line

```json
{
  "__meta__": {
    "dataset": "BASF-AI/ChemRxivRetrieval",
    "model": "BASF-AI/ChEmbed-prog",
    "top_k": 10,
    "n_success": 25,
    "n_fail": 25,
    "scanned_queries": 1234,
    "corpus_size": 69538,
    "notes": {
      "corpus_text": "title + ' ' + text (if title present) else text",
      "query_text": "queries['text'] as-is",
      "similarity": "dot product on L2-normalized embeddings (cosine)",
      "scope": "full corpus searched for every scanned query (no subsetting)",
      "memory_strategy": "chunked corpus encoding + running top-k merge",
      "query_batch_size": 16,
      "corpus_chunk_size": 2048,
      "max_seq_length": 512
    }
  }
}
```

## Example line

```json
{
  "query_id": "1383",
  "query_text": "How does ...?",
  "ground_truth_doc_id": "62fd8c3806e43b40c525369f_1",
  "ground_truth_score": 1,
  "ground_truth_text": "...",
  "top_k": 10,
  "retrieved": [
    {
      "rank": 1,
      "score": 0.7134,
      "doc_id": "...",
      "text": "..."
    }
  ],
  "success": true
}
```

`success=true` means `ground_truth_doc_id` appears in `retrieved` top-k.
