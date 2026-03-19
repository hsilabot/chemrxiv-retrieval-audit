#!/usr/bin/env python3
"""Generate ChemRxiv retrieval review packet with full-corpus search and memory-safe chunking.

Key guarantees:
- NO corpus subsetting: every scanned query is searched against the full corpus.
- Memory-safe: corpus encoded in chunks; similarity computed chunk-by-chunk.
- Running top-k maintained per query across all corpus chunks (never materialize query x corpus matrix).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


@dataclass
class RetrievedDoc:
    rank: int
    score: float
    doc_id: str
    text: str


@dataclass
class Example:
    query_id: str
    query_text: str
    ground_truth_doc_id: str
    ground_truth_score: int
    ground_truth_text: str
    top_k: int
    retrieved: List[RetrievedDoc]
    success: bool


def get_row_id(row: Dict[str, Any]) -> str:
    if "id" in row:
        return str(row["id"])
    if "_id" in row:
        return str(row["_id"])
    raise KeyError(f"Missing id/_id in row keys={list(row.keys())}")


def mteb_corpus_text(row: Dict[str, Any]) -> Tuple[str, str | None, str]:
    title = row.get("title")
    body = row.get("text") or ""
    if isinstance(title, str) and title:
        full = (title + " " + str(body)).strip()
    else:
        title = None
        full = str(body).strip()
    return full, title, str(body)


def pick_config(configs: List[str], want: str) -> str:
    if want in configs:
        return want
    if want == "qrels" and "default" in configs:
        return "default"
    raise RuntimeError(f"Config '{want}' not found. Available: {configs}")


def load_strict_split(dataset: str, config: str, split: str, revision: str | None):
    """Load dataset config for an exact split.

    Returns (dataset_split, used_split).
    """
    splits = get_dataset_split_names(dataset, config_name=config, revision=revision)
    if split not in splits:
        raise ValueError(
            f"Split '{split}' not available for config '{config}'. Available splits: {splits}"
        )
    return load_dataset(dataset, config, split=split, revision=revision), split


def split_for_config(requested_split: str, config: str) -> str:
    """ChemRxivRetrieval has non-uniform split names across configs.

    Required mapping (hard-coded by design):
    - corpus: train
    - queries: train
    - qrels (default): test

    If the dataset ever changes, this mapping should be updated intentionally.
    """
    if config in {"corpus", "queries"}:
        return "train"
    if config == "qrels":
        return "test"
    return requested_split


def batched(xs: List[Any], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="BASF-AI/ChemRxivRetrieval")
    ap.add_argument("--revision", default=None)
    # Hard-coded by design for reproducibility
    ap.add_argument("--split", default="test", help="(hard-coded default) dataset split")
    ap.add_argument("--model-id", default="BASF-AI/ChEmbed-prog")
    ap.add_argument("--success-target", type=int, default=25)
    ap.add_argument("--failure-target", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--query-batch-size", type=int, default=16)
    ap.add_argument("--corpus-chunk-size", type=int, default=2048)
    ap.add_argument("--max-seq-length", type=int, default=512)
    # Device string forwarded to SentenceTransformer (PyTorch):
    # - auto (default): let sentence-transformers pick (CUDA if available, else MPS on Apple, else CPU)
    # - cpu | mps | cuda | cuda:0 | ...
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-queries", type=int, default=None, help="Optional cap while searching for targets")
    ap.add_argument("--output-path", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Dataset loading
    configs = get_dataset_config_names(args.dataset, revision=args.revision)
    corpus_cfg = pick_config(configs, "corpus")
    queries_cfg = pick_config(configs, "queries")
    qrels_cfg = pick_config(configs, "qrels")

    # Split is strict by design (no fallback), but split name is config-specific.
    corpus_want = split_for_config(args.split, corpus_cfg)
    queries_want = split_for_config(args.split, queries_cfg)
    qrels_want = split_for_config(args.split, qrels_cfg)

    corpus_ds, corpus_split = load_strict_split(args.dataset, corpus_cfg, corpus_want, args.revision)
    queries_ds, queries_split = load_strict_split(args.dataset, queries_cfg, queries_want, args.revision)
    qrels_ds, qrels_split = load_strict_split(args.dataset, qrels_cfg, qrels_want, args.revision)

    print(f"Dataset: {args.dataset} (revision={args.revision})")
    print(f"Requested split: {args.split}")
    print(f"Corpus config/split: {corpus_cfg}/{corpus_split} (rows={len(corpus_ds)})")
    print(f"Queries config/split: {queries_cfg}/{queries_split} (rows={len(queries_ds)})")
    print(f"Qrels config/split: {qrels_cfg}/{qrels_split} (rows={len(qrels_ds)})")

    # qrels map
    qrels_map: Dict[str, Dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = int(row["score"])
        qrels_map.setdefault(qid, {})[cid] = score

    # Preload corpus metadata/text once (full corpus)
    corpus_ids: List[str] = []
    corpus_fulltext: List[str] = []
    corpus_titles: List[str | None] = []
    corpus_bodies: List[str] = []
    corpus_id_to_idx: Dict[str, int] = {}

    for i, row in enumerate(corpus_ds):
        cid = get_row_id(row)
        full, title, body = mteb_corpus_text(row)
        corpus_ids.append(cid)
        corpus_fulltext.append(full)
        corpus_titles.append(title)
        corpus_bodies.append(body)
        corpus_id_to_idx[cid] = i

    # Filter queries to those with qrels, then shuffle deterministically
    queries: List[Dict[str, Any]] = [q for q in queries_ds if get_row_id(q) in qrels_map]
    random.shuffle(queries)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    # Model setup
    st_kwargs: Dict[str, Any] = {
        # Required for BASF-AI/ChEmbed-prog (custom HF code)
        "trust_remote_code": True,
    }
    if args.device != "auto":
        st_kwargs["device"] = args.device

    model = SentenceTransformer(args.model_id, **st_kwargs)
    model.max_seq_length = args.max_seq_length


    # Build active query list (one GT per query)
    active_queries: List[Tuple[str, str, str, int]] = []
    for q in queries:
        qid = get_row_id(q)
        rels = qrels_map[qid]
        gt_doc_id, gt_score = max(rels.items(), key=lambda kv: kv[1])
        if gt_doc_id not in corpus_id_to_idx:
            continue
        qtext = str(q.get("text", q.get("query", "")))
        active_queries.append((qid, qtext, gt_doc_id, int(gt_score)))

    if not active_queries:
        raise RuntimeError("No queries with qrels+corpus match found.")

    # Encode all queries once (MTEB-style): then scan corpus chunks and update running top-k.
    print(f"Encoding queries: {len(active_queries)}")
    q_texts = [x[1] for x in active_queries]
    q_emb_parts: List[np.ndarray] = []
    for q_batch in tqdm(list(batched(q_texts, args.query_batch_size)), desc="Encoding queries", unit="batch"):
        q_emb_parts.append(
            model.encode(
                q_batch,
                batch_size=args.query_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
        )
    q_emb = np.vstack(q_emb_parts)

    n_queries = q_emb.shape[0]
    k = args.top_k
    best_scores = np.full((n_queries, k), -np.inf, dtype=np.float32)
    best_indices = np.full((n_queries, k), -1, dtype=np.int64)

    started = time.time()

    # Encode corpus in chunks ONCE; compute similarities against ALL queries; maintain running top-k per query.
    n_chunks = math.ceil(len(corpus_fulltext) / args.corpus_chunk_size)
    p_chunks = tqdm(
        range(0, len(corpus_fulltext), args.corpus_chunk_size),
        total=n_chunks,
        desc="Corpus chunks (encode+score)",
        unit="chunk",
    )
    for c_start in p_chunks:
        c_end = min(c_start + args.corpus_chunk_size, len(corpus_fulltext))
        c_texts = corpus_fulltext[c_start:c_end]
        c_emb = model.encode(
            c_texts,
            batch_size=max(8, min(128, args.corpus_chunk_size // 8)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores = q_emb @ c_emb.T  # [Q, C]
        local_idx = np.arange(c_start, c_end, dtype=np.int64)

        merged_scores = np.concatenate([best_scores, scores], axis=1)
        repeated_local = np.broadcast_to(local_idx, (n_queries, local_idx.shape[0]))
        merged_indices = np.concatenate([best_indices, repeated_local], axis=1)

        part = np.argpartition(-merged_scores, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        picked_scores = merged_scores[rows, part]
        picked_indices = merged_indices[rows, part]

        order = np.argsort(-picked_scores, axis=1)
        best_scores = picked_scores[rows, order]
        best_indices = picked_indices[rows, order]

        elapsed = time.time() - started
        p_chunks.set_postfix(elapsed=f"{elapsed/60:.1f}m")

    scanned_queries = len(active_queries)

    # Select packet examples AFTER full retrieval (since success depends on full-corpus top-k).
    success_examples: List[Example] = []
    failure_examples: List[Example] = []

    shuffled = list(range(len(active_queries)))
    random.shuffle(shuffled)

    for i in shuffled:
        if len(success_examples) >= args.success_target and len(failure_examples) >= args.failure_target:
            break

        qid, qtext, gt_doc_id, gt_score = active_queries[i]
        top_doc_indices = best_indices[i].tolist()
        top_scores = best_scores[i].tolist()
        retrieved: List[RetrievedDoc] = []

        for rank, (doc_idx, score) in enumerate(zip(top_doc_indices, top_scores), start=1):
            if doc_idx < 0:
                continue
            retrieved.append(
                RetrievedDoc(
                    rank=rank,
                    score=float(score),
                    doc_id=corpus_ids[doc_idx],
                    text=corpus_bodies[doc_idx],
                )
            )

        success = any(r.doc_id == gt_doc_id for r in retrieved)
        gt_idx = corpus_id_to_idx[gt_doc_id]
        example = Example(
            query_id=qid,
            query_text=qtext,
            ground_truth_doc_id=gt_doc_id,
            ground_truth_score=gt_score,
            ground_truth_text=corpus_bodies[gt_idx],
            top_k=args.top_k,
            retrieved=retrieved,
            success=success,
        )

        if success and len(success_examples) < args.success_target:
            success_examples.append(example)
        elif (not success) and len(failure_examples) < args.failure_target:
            failure_examples.append(example)

    print(f"Selected success examples: {len(success_examples)} / {args.success_target}")
    print(f"Selected failure examples: {len(failure_examples)} / {args.failure_target}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        meta = {
            "dataset": args.dataset,
            "revision": args.revision,
            "split": args.split,
            "model": args.model_id,
            "seed": args.seed,
            "top_k": args.top_k,
            # Match prior packet style
            "n_success": len(success_examples),
            "n_fail": len(failure_examples),
            "scanned_queries": scanned_queries,
            "corpus_size": len(corpus_ids),
            "notes": {
                "corpus_text": "title + ' ' + text (if title present) else text",
                "query_text": "queries['text'] as-is",
                "similarity": "dot product on L2-normalized embeddings (cosine)",
                "scope": "full corpus searched for every scanned query (no subsetting)",
                "memory_strategy": "MTEB-style: encode queries once + encode corpus in chunks + running top-k merge",
                "query_batch_size": args.query_batch_size,
                "corpus_chunk_size": args.corpus_chunk_size,
                "max_seq_length": args.max_seq_length,
                "device": args.device,
            },
        }
        f.write(json.dumps({"__meta__": meta}, ensure_ascii=False) + "\n")
        for ex in success_examples + failure_examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    print(f"Wrote: {args.output_path}")
    print(f"Success examples: {len(success_examples)} / {args.success_target}")
    print(f"Failure examples: {len(failure_examples)} / {args.failure_target}")
    print(f"Scanned queries: {scanned_queries}")
    print(f"Corpus size: {len(corpus_ids)}")


if __name__ == "__main__":
    main()
