"""
5 版评测结果对比可视化
读取 eval_output/final_*_summary.json 以及 eval_output/ood_correctness.json，
生成 4 张 PNG 到 docs/figures/：

  01_overall_radar.png        — 主链雷达（in-domain 100 题，6 核心指标，5 版）
  02_latency.png              — 主链延迟 P50/P95（5 版）
  03_routing_ablation.png     — 路由专项（准确率 + P50 延迟，3 版对比）
  04_crag_ood.png             — CRAG OOD 价值（LLM 判分正确性，5 版）

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

# 5 版主链（按进阶顺序，颜色由浅灰→深绿）
VERSIONS = [
    ("final_baseline",  "Baseline\n(round-robin+纯LLM路由)", "#9aa0a6"),
    ("final_retrieval", "+BM25/RRF",                          "#e9c46a"),
    ("final_parentdoc", "+父文档回填",                        "#f4a261"),
    ("final_route",     "+智能路由(function-call)",           "#e76f51"),
    ("final_crag",      "+CRAG网络检索",                      "#2a9d8f"),
]

# 路由专项（受控变量：固定 BM25/RRF+父文档、CRAG off，只换路由器）
ROUTERS = [
    ("final_parentdoc",  "纯LLM(原始)",       "#9aa0a6"),
    ("final_route_rule", "D3规则(规则短路)",  "#f4a261"),
    ("final_route",      "function-call",     "#2a9d8f"),
]


def load_summary(run_id: str) -> dict:
    p = EVAL_DIR / f"{run_id}_summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"未找到 {p}，请先跑评测：python -m eval.eval_runner --run_id {run_id}"
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_ood_correctness() -> dict:
    p = EVAL_DIR / "ood_correctness.json"
    if not p.exists():
        raise FileNotFoundError(f"未找到 {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------- 图1：主链雷达（in-domain）----------
def plot_radar(summaries: list[dict]) -> Path:
    metrics = [
        ("Hit@5",            lambda s: s["aggregated_in_domain"]["retrieval"]["hit@5"]),
        ("Recall@5",         lambda s: s["aggregated_in_domain"]["retrieval"]["recall@5"]),
        ("MRR@10",           lambda s: s["aggregated_in_domain"]["retrieval"]["mrr@10"]),
        ("路由准确率",        lambda s: s["aggregated_in_domain"]["routing"]["accuracy"]),
        ("Faithfulness",     lambda s: s["aggregated_in_domain"]["generation"]["faithfulness"]),
        ("Answer Relevancy", lambda s: s["aggregated_in_domain"]["generation"]["answer_relevancy"]),
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
        "in-domain 100 题 · Baseline → +BM25/RRF → +父文档 → +智能路由 → +CRAG 累加\n总体核心指标对比",
        fontsize=12, pad=22
    )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        fontsize=9, ncol=2, frameon=False
    )
    plt.tight_layout()
    out = FIG_DIR / "01_overall_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------- 图2：主链延迟（in-domain）----------
def plot_latency(summaries: list[dict]) -> Path:
    metric_keys = ["latency_p50_ms", "latency_p95_ms"]
    metric_labels = ["P50（中位）", "P95"]
    n_versions = len(VERSIONS)
    width = 0.14
    x = np.arange(len(metric_keys))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (s, (_, vlabel, color)) in enumerate(zip(summaries, VERSIONS)):
        values = [s["aggregated_in_domain"]["system"][k] for k in metric_keys]
        # 居中分布：offset 以 n_versions 为中心
        offset = (i - (n_versions - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width, label=vlabel,
            color=color, edgecolor="white", linewidth=0.6
        )
        max_val = max(
            s["aggregated_in_domain"]["system"]["latency_p95_ms"] for s in summaries
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.012,
                f"{v / 1000:.1f}s",
                ha="center", fontsize=8.5
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("延迟 (ms)", fontsize=11)
    ax.set_title(
        "端到端延迟对比（in-domain）— 智能路由 fast path 是延迟下降主因",
        fontsize=12, pad=10
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    max_p95 = max(s["aggregated_in_domain"]["system"]["latency_p95_ms"] for s in summaries)
    ax.set_ylim(0, max_p95 * 1.20)

    plt.tight_layout()
    out = FIG_DIR / "02_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------- 图3：路由专项（受控变量）----------
def plot_routing_ablation(router_summaries: list[dict]) -> Path:
    fig, ax1 = plt.subplots(figsize=(8, 6))

    x = np.arange(len(ROUTERS))
    width = 0.45

    accuracies = [
        s["aggregated"]["routing"]["accuracy"] for s in router_summaries
    ]
    p50s_s = [
        s["aggregated"]["system"]["latency_p50_ms"] / 1000 for s in router_summaries
    ]
    labels = [r[1] for r in ROUTERS]
    colors = [r[2] for r in ROUTERS]

    bars = ax1.bar(x, accuracies, width, color=colors, edgecolor="white", linewidth=0.8)

    # 柱顶标准确率数值
    for bar, acc in zip(bars, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    # 柱下方附 P50 延迟
    for i, p50 in enumerate(p50s_s):
        ax1.text(
            x[i],
            -0.06,
            f"P50={p50:.1f}s",
            ha="center", va="top", fontsize=9.5, color="#444444",
            transform=ax1.get_xaxis_transform()
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel("路由准确率", fontsize=11)
    ax1.set_ylim(0, 1.18)
    ax1.set_title(
        "路由专项（受控变量）：准确率 0.758 → 1.0，延迟下降\n"
        "固定 BM25/RRF + 父文档，仅换路由器",
        fontsize=12, pad=10
    )
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "03_routing_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ---------- 图4：CRAG OOD 价值（头条）----------
def plot_crag_ood(ood_data: dict) -> Path:
    llm_judge = ood_data["llm_judge"]
    # 按主链 5 版顺序
    run_ids = [v[0] for v in VERSIONS]
    labels = [v[1] for v in VERSIONS]
    # CRAG 柱用强调绿色，其余灰色
    highlight_color = "#2a9d8f"
    default_color = "#b0b8c1"
    colors = [
        highlight_color if rid == "final_crag" else default_color
        for rid in run_ids
    ]
    values = [llm_judge[rid] for rid in run_ids]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(run_ids))
    bars = ax.bar(x, values, 0.55, color=colors, edgecolor="white", linewidth=0.8)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("OOD 答案正确性（LLM 判分）", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_title(
        "库外(OOD)问题答案正确性(LLM判分)\n"
        "不开 CRAG 几乎全拒答(0.03–0.19) → CRAG 0.83",
        fontsize=12, pad=10
    )
    ax.grid(axis="y", alpha=0.3)

    # 标注 CRAG 关键说明
    crag_idx = run_ids.index("final_crag")
    ax.annotate(
        "CRAG 触发网络检索\n(20/20 OOD 题全覆盖)",
        xy=(crag_idx, values[crag_idx]),
        xytext=(crag_idx - 1.2, values[crag_idx] + 0.08),
        fontsize=9,
        color="#2a9d8f",
        arrowprops=dict(arrowstyle="->", color="#2a9d8f", lw=1.5),
    )

    plt.tight_layout()
    out = FIG_DIR / "04_crag_ood.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    # 加载主链 5 版
    summaries = [load_summary(v[0]) for v in VERSIONS]
    print(f"已读取 5 版主链数据：{[v[0] for v in VERSIONS]}")

    # 加载路由专项 3 版
    router_summaries = [load_summary(r[0]) for r in ROUTERS]
    print(f"已读取路由专项数据：{[r[0] for r in ROUTERS]}")

    # 加载 OOD 数据
    ood_data = load_ood_correctness()
    print("已读取 OOD 正确性数据")

    # 删除旧孤儿图
    old_fig = FIG_DIR / "02_mrr_by_question_type.png"
    if old_fig.exists():
        old_fig.unlink()
        print(f"  已删除旧图 → {old_fig.relative_to(ROOT)}")

    # 生成 4 图
    for fn, args in [
        (plot_radar,            (summaries,)),
        (plot_latency,          (summaries,)),
        (plot_routing_ablation, (router_summaries,)),
        (plot_crag_ood,         (ood_data,)),
    ]:
        out = fn(*args)
        print(f"  saved → {out}")

    print(f"\n所有图保存在 {FIG_DIR}/")


if __name__ == "__main__":
    main()
