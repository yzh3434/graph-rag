"""离线计算"答案正确性 vs ground_truth"（embedding 余弦），按 domain 分组求均值。
不重跑评测——直接读 per_sample.jsonl 的 answer + testset 的 ground_truth。

用法：
  python -m eval.score_answer_correctness \
      --per_sample eval_output/final_crag_per_sample.jsonl \
      --testset testset_output/testset.jsonl
"""
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Callable


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_records(records: List[Dict], ground_truths: List[str],
                  embed_fn: Callable[[str], List[float]]) -> Dict[str, float]:
    """records 与 ground_truths 一一对应（同序）。返回 {domain: mean_cosine}。"""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for rec, gt in zip(records, ground_truths):
        ans = (rec.get("answer") or "").strip()
        gt = (gt or "").strip()
        domain = rec.get("domain", "in_domain")
        if not ans or not gt:
            continue
        buckets[domain].append(_cosine(embed_fn(ans), embed_fn(gt)))
    return {d: (sum(v) / len(v) if v else 0.0) for d, v in buckets.items()}


def _load_embed_fn():
    """复用项目 embedding（BGE-small-zh）。"""
    from rag_modules.milvus_index_construction import MilvusIndexConstructionModule
    mod = MilvusIndexConstructionModule()
    return lambda s: mod.embeddings.embed_query(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_sample", required=True)
    ap.add_argument("--testset", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.per_sample, encoding="utf-8")]
    testset = [json.loads(l) for l in open(args.testset, encoding="utf-8")]
    # per_sample 有 sample_idx，按它对齐 testset 顺序
    gts = [testset[r["sample_idx"]].get("ground_truth", "") for r in records]

    embed_fn = _load_embed_fn()
    result = score_records(records, gts, embed_fn)
    print("答案正确性（vs ground_truth，embedding 余弦）按 domain：")
    for d, v in sorted(result.items()):
        print(f"  {d}: {v:.4f}")


if __name__ == "__main__":
    main()
