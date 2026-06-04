"""LLM 判分答案正确性模块。

对照 ground_truth，用 LLM 给系统 answer 打 [0, 1] 分：
  - 拒答（"没有找到 / 无法回答"等）→ 0
  - 答案正确且覆盖 ground_truth 关键信息 → 接近 1

适用场景：评测"开 / 不开 CRAG 网络检索"在库外（OOD）问题上的答题质量，
embedding 余弦在拒答上区分度不足（拒答也能达 0.86），本模块提供更高区分度的判分。

用法（CLI）：
  python -m eval.score_llm_correctness \\
      --per_sample eval_output/final_crag_per_sample.jsonl \\
      --testset testset_output/testset.jsonl \\
      --domain out_of_domain

用法（程序内注入假 judge）：
  from eval.score_llm_correctness import score_records_llm
  result = score_records_llm(records, questions, ground_truths, judge_fn=my_fn)
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 核心评分函数（可注入 judge_fn，便于单测不调真实 LLM）
# ---------------------------------------------------------------------------

def score_records_llm(
    records: List[Dict],
    questions: List[str],
    ground_truths: List[str],
    judge_fn: Callable[[str, str, str], float],
) -> Dict[str, float]:
    """对一批记录用 judge_fn 打分，按 domain 分桶返回均值。

    参数
    ----
    records        : 每条含 ``answer``（str）、``domain``（str，缺省 "in_domain"）
    questions      : 与 records 一一对应的问题文本列表
    ground_truths  : 与 records 一一对应的参考答案列表
    judge_fn       : ``(question, answer, ground_truth) -> float``，返回 [0, 1] 分

    返回
    ----
    ``{domain: mean_score}``，跳过 answer 或 ground_truth 为空的样本。
    """
    buckets: Dict[str, List[float]] = defaultdict(list)

    for rec, q, gt in zip(records, questions, ground_truths):
        ans = (rec.get("answer") or "").strip()
        gt_str = (gt or "").strip()
        domain = rec.get("domain", "in_domain")

        if not ans or not gt_str:
            continue

        score = judge_fn(q, ans, gt_str)
        buckets[domain].append(float(score))

    return {d: (sum(v) / len(v) if v else 0.0) for d, v in buckets.items()}


# ---------------------------------------------------------------------------
# 真实 judge（使用 DeepSeek，与 generation_integration.py 保持一致）
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
你是一位专业的烹饪领域问答评测专家。
你的任务是：对照参考答案（ground_truth），判断系统给出的 answer 是否正确。

评分规则：
1. 如果 answer 表示"无法回答"、"没有相关信息"、"抱歉找不到"等拒答，直接给 0 分。
2. 如果 answer 的核心内容（食材、步骤、方法、说法）与 ground_truth 一致或覆盖了关键信息，给接近 1 的分数。
3. 如果 answer 部分正确（步骤缺失、食材遗漏、有误导性内容），酌情给 0.3～0.7 之间的分数。
4. 如果 answer 明显错误或与 ground_truth 完全不符，给 0 分。

请严格只返回如下 JSON，不要附加任何解释：
{"score": <0到1之间的小数，保留2位小数>}
"""

_JUDGE_USER_TEMPLATE = """\
问题：{question}

参考答案（ground_truth）：
{ground_truth}

系统回答（answer）：
{answer}

请按评分规则打分，严格返回 JSON：{{"score": ...}}
"""


def _default_judge_fn() -> Callable[[str, str, str], float]:
    """构造并返回真实 DeepSeek judge 闭包。调用前需确保 .env 含 DEEPSEEK_API_KEY。"""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/",
    )

    def judge(question: str, answer: str, ground_truth: str) -> float:
        user_msg = _JUDGE_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
        )
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
            return float(data.get("score", 0.0))
        except Exception:
            return 0.0

    return judge


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="离线 LLM 判分答案正确性（对照 ground_truth，OOD 高区分度）"
    )
    ap.add_argument("--per_sample", required=True,
                    help="eval_output/<run>_per_sample.jsonl 路径")
    ap.add_argument("--testset", required=True,
                    help="testset_output/testset.jsonl 路径")
    ap.add_argument("--domain", default=None,
                    help="只评指定 domain（如 out_of_domain），留空则评全部")
    args = ap.parse_args()

    # 读取数据
    records: List[Dict] = [
        json.loads(line) for line in open(args.per_sample, encoding="utf-8")
    ]
    testset: List[Dict] = [
        json.loads(line) for line in open(args.testset, encoding="utf-8")
    ]

    # 按 sample_idx 对齐 question / ground_truth
    questions: List[str] = [
        testset[r["sample_idx"]].get("question", "") for r in records
    ]
    gts: List[str] = [
        testset[r["sample_idx"]].get("ground_truth", "") for r in records
    ]

    # 可选：只保留指定 domain
    if args.domain:
        filtered = [
            (rec, q, gt)
            for rec, q, gt in zip(records, questions, gts)
            if rec.get("domain", "in_domain") == args.domain
        ]
        records, questions, gts = (
            [x[0] for x in filtered],
            [x[1] for x in filtered],
            [x[2] for x in filtered],
        )

    judge_fn = _default_judge_fn()
    result = score_records_llm(records, questions, gts, judge_fn)

    print("LLM 判分答案正确性（对照 ground_truth）按 domain：")
    for domain, score in sorted(result.items()):
        count = sum(
            1 for rec in records
            if (rec.get("domain", "in_domain") == domain
                and (rec.get("answer") or "").strip()
                and gts[records.index(rec)])
        )
        print(f"  {domain}: {score:.4f}  (n={count})")


if __name__ == "__main__":
    main()
