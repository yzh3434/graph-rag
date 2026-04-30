"""
三版评测结果对比可视化
读取 eval_output/{baseline_v2,route_v2,retrieval_v1}_summary.json，
画三张 PNG 到 docs/figures/，README 引用：

  01_overall_radar.png            — 总体雷达图（6 核心指标）
  02_mrr_by_question_type.png     — 按问题类型 MRR@10 柱状图
  03_latency.png                  — 端到端延迟 P50/P95 对比

用法（从项目根目录）：
    python -m eval.plot_comparison
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows / macOS / Linux 三平台 fallback）
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC", "Heiti TC", "Noto Sans CJK SC", "sans-serif"
]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval_output"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 三个版本（按时间顺序），颜色：灰 → 橙 → 绿（从浅到深暗示进阶）
VERSIONS = [
    ("baseline_v2",   "Baseline\n(占位 BM25 + Round-robin + 纯 LLM 路由)", "#9aa0a6"),
    ("route_v2",      "+ 智能路由两层架构 + Fast Path",                  "#f4a261"),
    ("retrieval_v1",  "+ 真 BM25(jieba) + RRF（最终版）",                 "#2a9d8f"),
]

# 7 类问题，前 5 类走 hybrid_traditional（D4 影响范围），后 2 类走 graph_rag/combined
QUESTION_TYPES = [
    ("simple_fact",     "事实查询"),
    ("attribute_query", "属性查询"),
    ("step_by_step",    "步骤型"),
    ("causal",          "因果推理"),
    ("entity_relation", "实体关系"),
    ("multi_hop",       "多跳查询"),
    ("comparison",      "两菜对比"),
]
N_HYBRID = 5  # 前 5 类


def load_summary(run_id: str) -> dict:
    p = EVAL_DIR / f"{run_id}_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"未找到 {p}，请先跑评测：python -m eval.eval_runner --run_id {run_id}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def plot_radar(summaries: list[dict]) -> Path:
    metrics = [
        ("Hit@5",            lambda s: s["aggregated"]["retrieval"]["hit@5"]),
        ("Recall@5",         lambda s: s["aggregated"]["retrieval"]["recall@5"]),
        ("MRR@10",           lambda s: s["aggregated"]["retrieval"]["mrr@10"]),
        ("路由准确率",        lambda s: s["aggregated"]["routing"]["accuracy"]),
        ("Faithfulness",     lambda s: s["aggregated"]["generation"]["faithfulness"]),
        ("Answer Relevancy", lambda s: s["aggregated"]["generation"]["answer_relevancy"]),
    ]
    labels = [m[0] for m in metrics]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    for s, (_, vlabel, color) in zip(summaries, VERSIONS):
        values = [m[1](s) for m in metrics]
        values += [values[0]]
        ax.plot(angles, values, color=color, linewidth=2.2, label=vlabel)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_rlim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_title(
        "总体核心指标对比\nBaseline → + 智能路由模块 → + BM25/RRF",
        fontsize=13, pad=22
    )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        fontsize=9, ncol=1, frameon=False
    )
    plt.tight_layout()
    out = FIG_DIR / "01_overall_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_mrr_by_type(summaries: list[dict]) -> Path:
    types = [t[0] for t in QUESTION_TYPES]
    type_labels = [t[1] for t in QUESTION_TYPES]
    n_types = len(types)
    width = 0.26
    x = np.arange(n_types)

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    for i, (s, (_, vlabel, color)) in enumerate(zip(summaries, VERSIONS)):
        values = [s["by_question_type"][t]["mrr@10"] for t in types]
        offset = (i - 1) * width  # 居中分布
        bars = ax.bar(x + offset, values, width, label=vlabel, color=color, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.2f}", ha="center", fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, fontsize=11)
    ax.set_ylabel("MRR@10", fontsize=11)
    ax.set_ylim(0, 1.20)
    ax.set_title(
        "按问题类型 MRR@10 对比 — BM25+RRF 仅作用于 hybrid_traditional 路径",
        fontsize=13, pad=10
    )
    # 图例放左上，避开右上区间标注
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # 分割线 + 区间标注（放在 1.13 高度，避开图例）
    sep_x = N_HYBRID - 0.5
    ax.axvline(x=sep_x, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(
        (N_HYBRID - 1) / 2, 1.13,
        "走 hybrid_traditional 路径（BM25/RRF 改进作用区）",
        ha="center", fontsize=9.5, color="#2a9d8f", fontweight="bold"
    )
    ax.text(
        N_HYBRID + (n_types - N_HYBRID - 1) / 2, 1.13,
        "走 graph_rag / combined（BM25/RRF 不作用）",
        ha="center", fontsize=9.5, color="#666666"
    )

    plt.tight_layout()
    out = FIG_DIR / "02_mrr_by_question_type.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_latency(summaries: list[dict]) -> Path:
    metric_keys = ["latency_p50_ms", "latency_p95_ms"]
    metric_labels = ["P50（中位）", "P95"]
    width = 0.26
    x = np.arange(len(metric_keys))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for i, (s, (_, vlabel, color)) in enumerate(zip(summaries, VERSIONS)):
        values = [s["aggregated"]["system"][k] for k in metric_keys]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=vlabel, color=color, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.012,
                f"{v / 1000:.1f}s", ha="center", fontsize=9.5
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("延迟 (ms)", fontsize=11)
    ax.set_title(
        "端到端延迟对比 — 智能路由 fast path 是延迟下降主要来源",
        fontsize=13, pad=10
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(s["aggregated"]["system"]["latency_p95_ms"] for s in summaries) * 1.15)

    plt.tight_layout()
    out = FIG_DIR / "03_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    summaries = [load_summary(v[0]) for v in VERSIONS]
    print(f"已读取三版数据：{[v[0] for v in VERSIONS]}")
    for fn in (plot_radar, plot_mrr_by_type, plot_latency):
        out = fn(summaries)
        print(f"  saved → {out.relative_to(ROOT)}")
    print(f"\n所有图保存在 {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
