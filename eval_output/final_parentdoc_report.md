# Eval Report — `final_parentdoc`

- **时间**：2026-06-03T21:16:21
- **样本数**：120 / 120（失败 0）
- **配置**：
  ```json
  {"use_bm25": false, "use_rrf": false, "use_rerank": false, "topic_alignment": false, "retrieve_top_k": 10, "embedding_model": "BAAI/bge-small-zh-v1.5", "llm_model": "deepseek-chat", "enable_crag": false, "enable_parent_doc_retrieval": true, "enable_bm25_rrf": true, "router": "pure_llm", "skip_generation_eval": false, "note": ""}
  ```

## 总体指标

| 层 | 指标 | 值 |
|---|---|---|
| 检索 | Hit@5 | 0.8000 |
| 检索 | Recall@5 | 0.7115 |
| 检索 | MRR@10 | 0.7611 |
| 路由 | Routing Accuracy | 0.7583 |
| 生成 | Faithfulness | 0.8045 |
| 生成 | Answer Relevancy | 0.8965 |
| 系统 | Latency P50 (ms) | 7475.7750 |
| 系统 | Latency P95 (ms) | 10962.4355 |

## 按 domain 分组（in-domain vs out-of-domain）

| domain | n | Hit@5 | Recall@5 | MRR@10 | 路由准确 | Faith | AR | 延迟均值(ms) |
|---|---|---|---|---|---|---|---|---|
| in_domain | 100 | 0.9600 | 0.8538 | 0.9133 | 0.7100 | 0.8498 | 0.9043 | 7621.9978 |
| out_of_domain | 20 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5705 | 0.8575 | 8262.1635 |

## 按问题类型分组

| 类型 | n | Hit@5 | Recall@5 | MRR@10 | 路由准确 | Faith | AR | 延迟均值(ms) |
|---|---|---|---|---|---|---|---|---|
| step_by_step | 25 | 0.6000 | 0.6000 | 0.5800 | 1.0000 | 0.7898 | 0.8824 | 9392.0228 |
| multi_hop | 15 | 0.9333 | 0.2250 | 0.7556 | 0.0000 | 0.8222 | 0.8943 | 8677.5060 |
| entity_relation | 25 | 0.6000 | 0.6000 | 0.6000 | 1.0000 | 0.7532 | 0.8949 | 6534.2404 |
| attribute_query | 15 | 1.0000 | 1.0000 | 0.9667 | 1.0000 | 0.9537 | 0.9368 | 5649.4547 |
| causal | 15 | 0.8000 | 0.8000 | 0.8000 | 0.7333 | 0.4989 | 0.8882 | 9429.0907 |
| simple_fact | 15 | 1.0000 | 1.0000 | 0.9333 | 1.0000 | 0.9056 | 0.8760 | 5896.2660 |
| comparison | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9627 | 0.9220 | 8450.1710 |

## 按预期策略分组

| 策略 | n | Hit@5 | Recall@5 | MRR@10 | 路由准确 |
|---|---|---|---|---|---|
| hybrid_traditional | 95 | 0.7579 | 0.7579 | 0.7368 | 0.9579 |
| graph_rag | 15 | 0.9333 | 0.2250 | 0.7556 | 0.0000 |
| combined | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Bad Cases — Top 10 最差 MRR

- **[causal / 预期 hybrid_traditional → 实际 graph_rag]** MRR=0.0000
  - 问题：为什么红烧鲤鱼要加入五花肉？
  - 预期 node_ids：['201000127']
  - 检索 node_ids：[]
- **[causal / 预期 hybrid_traditional → 实际 graph_rag]** MRR=0.0000
  - 问题：为什么柱候牛腩要先用中小火炒冰糖至融化再炒牛肉？
  - 预期 node_ids：['201002733']
  - 检索 node_ids：[]
- **[causal / 预期 hybrid_traditional → 实际 graph_rag]** MRR=0.0000
  - 问题：为什么粉蒸肉要腌制后再裹蒸肉米粉？
  - 预期 node_ids：['201001891']
  - 检索 node_ids：[]
- **[step_by_step / 预期 hybrid_traditional → 实际 hybrid_traditional]** MRR=0.0000
  - 问题：惠灵顿牛排的制作步骤是什么？
  - 预期 node_ids：[]
  - 检索 node_ids：['201002876', '201004544', '201003658', '201002512', '201004806', '201003314', '201000006', '201002797', '201001147', '201002391']
- **[step_by_step / 预期 hybrid_traditional → 实际 hybrid_traditional]** MRR=0.0000
  - 问题：如何制作法式洋葱汤？请分步骤说明。
  - 预期 node_ids：[]
  - 检索 node_ids：['201003507', '201004215', '201003931', '201003683', '201000272', '201004525', '201004040', '201005435', '201000815', '201001727']
- **[step_by_step / 预期 hybrid_traditional → 实际 hybrid_traditional]** MRR=0.0000
  - 问题：西班牙海鲜饭的制作步骤是什么？
  - 预期 node_ids：[]
  - 检索 node_ids：['201004544', '201000272', '201000395', '201004282', '201004466', '201000496', '201004588', '201000001', '201002555', '201001147']
- **[step_by_step / 预期 hybrid_traditional → 实际 hybrid_traditional]** MRR=0.0000
  - 问题：班尼迪克蛋的制作步骤是什么？
  - 预期 node_ids：[]
  - 检索 node_ids：['201000922', '201000519', '201000744', '201000730', '201000755', '201000999', '201000579', '201001122', '201004525', '201000006']
- **[step_by_step / 预期 hybrid_traditional → 实际 hybrid_traditional]** MRR=0.0000
  - 问题：如何制作意式千层面？请分步骤说明。
  - 预期 node_ids：[]
  - 检索 node_ids：['201004466', '201004215', '201004172', '201004353', '201004183', '201004040', '201004076', '201004746', '201004467', '201001147']
- **[step_by_step / 预期 hybrid_traditional → 实际 hybrid_traditional]** MRR=0.0000
  - 问题：韩式炸鸡的制作步骤是什么？
  - 预期 node_ids：[]
  - 检索 node_ids：['201003275', '201004801', '201002575', '201001727', '201001685', '201005596', '201000395', '201000386', '201000411', '201001147']
- **[step_by_step / 预期 hybrid_traditional → 实际 hybrid_traditional]** MRR=0.0000
  - 问题：如何制作冬阴功汤？请分步骤说明。
  - 预期 node_ids：[]
  - 检索 node_ids：['201003707', '201003048', '201003683', '201000496', '201004002', '201003931', '201000272', '201003873', '201001147', '201000160']
