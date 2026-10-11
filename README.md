# LitScope

输入一个研究主题，得到一份每个结论都能回溯到原文的工作草稿。

## 解决什么问题

做综述时，检索、去重、归纳、标注出处是四件分离的手工活。LitScope 把它们串成一条可评测的流水线。

## 架构

    主题 -> 检索（arXiv / Crossref / PubMed）-> 去重归一化
         -> 分块嵌入 -> 向量检索 -> LLM 生成 -> 带引用答案
         -> 评测（Recall@k / 引用支持率 / 裁判负对照 / 延迟 / 成本）

详细设计见 docs/superpowers/specs/2026-10-12-litscope-design.md

## 快速开始

    python -m venv .venv
    .venv/Scripts/activate
    pip install -r requirements.txt

    # 把 DeepSeek key 填进 .env（见 .env.example）
    python scripts/check_env.py
    python scripts/ping_llm.py

    # 多源检索 + 去重
    python scripts/search_demo.py 'graph neural network' --limit 5

    # 端到端问答（自动建语料 -> 建索引 -> 带引用回答）
    python scripts/ask.py --topic 'graph neural network' --papers 6 \
        --question '图神经网络靠什么机制传播信息？'

    # 评测
    python eval/build_eval_set.py --topic 'language model' --count 12
    python eval/run_eval.py --limit 10 --sleep 1.0
    python eval/run_rag_eval.py --topic 'graph neural network' --questions 6
    python eval/judge_sanity.py

## 量化结果

### 1. 检索阶段（多源召回 + 去重）

| 指标 | 值 |
| --- | --- |
| Recall@1 | 0.9167 |
| Recall@5 | 1.0000 |
| Recall@10 | 1.0000 |
| MRR | 0.9583 |
| 延迟 P50 / P95 | 7.06 s / 10.30 s |
| 评测条数 | 12（失败 0） |

### 2. RAG 端到端（6 个问题，top_k=5）

| 指标 | 值 |
| --- | --- |
| 目标论文检索率 | 1.0000 |
| 引用支持率（LLM 裁判） | 1.0000（12 对） |
| 无效引用标记数 | 0 |
| 延迟 P50 / P95 | 1.62 s / 1.99 s |
| token 总数 / 均值 | 3922 / 653.7 |

### 3. 裁判负对照（判官可信度检查）

让同一个 claim 去配**另一篇论文**的段落，看裁判会不会照单全收：

| 指标 | 值 |
| --- | --- |
| 正例支持率（claim + 自己的来源） | 1.0000（6/6） |
| 负例拒绝率（claim + 别人的来源） | 1.0000（6/6） |
| **平衡准确率** | **1.0000** |

平衡准确率 0.5 意味着裁判是橡皮图章（永远说 SUPPORTED），1.0 意味着完美区分。所以上面那个 1.0 的引用支持率不是白来的——**裁判确实会拒绝**。

原始结果：metrics.json、metrics_rag.json、metrics_judge_sanity.json。

## Web 服务

    pip install -r requirements.txt
    python -m uvicorn app.api.app:app --reload --port 8000

打开 http://127.0.0.1:8000 就是一个可用的界面（FastAPI 直接托管，无需构建步骤）。

| 接口 | 说明 |
| --- | --- |
| GET / | 内置 Web UI |
| GET /api/health | 健康检查，返回嵌入器与 LLM 状态 |
| GET /api/search?q=&limit=&sources= | 多源检索 + 去重 |
| POST /api/ask | {topic, question, top_k, per_source} -> 带引用回答 |

语料按 topic 缓存，同一主题第二次提问不会重复检索。

实测（真实网络 + DeepSeek）：

    HEALTH 200 {status: ok, embedder: hashing, llm: ready}
    GET  /             -> 200, 4117 bytes
    GET  /api/search   -> 200, raw 9, deduped 9, errors []
    POST /api/ask      -> 200, corpus 12 papers / 46 chunks,
                          tokens 515, latency 0.79s, valid citations 4, invalid 0

注意最后那条回答**拒绝了编造**：它明说现有来源未说明通用传播机制，并逐条说明每个来源实际讲的是什么。这是带引用 RAG 应有的行为，也是这个项目最值得演示的一段。

## 部署

计划：

    Railway / Render  <- FastAPI（含内置 UI），环境变量里配 DEEPSEEK_API_KEY
    Vercel            <- 可选：Next.js 前端

**当前阻塞**：这台机器上没有 git / node / npm（已确认），所以：

1. 要推 GitHub、部署到任何平台，先装 Git for Windows：https://git-scm.com/download/win
2. 要做 Next.js 前端，还要装 Node.js LTS：https://nodejs.org/

在装好之前，app/api/index.html 这个内置 UI 已经能覆盖「可在线访问」这条硬要求，而且是单服务部署，最简单。

## 这些数字的口径（重要）

诚实说明，避免过度解读：

1. **评测标签偏简单**。检索评测用的是 title-recovery（拿论文标题当 query）；RAG 评测的问题里也直接包含论文标题。标题几乎是最强查询，所以 Recall@5 = 1.0、目标检索率 1.0 属于**预期之内**，它证明链路正确，**不代表语义检索质量**。
2. **样本量小**。6 题 / 12 对引用，统计意义有限，置信区间很宽。
3. **裁判与生成同源**（都是 deepseek-flash），存在自我偏好风险。负对照只做了部分缓解，不能完全排除。
4. **嵌入是词面哈希**（HashingEmbedder），检索得分只有 0.06–0.09 量级，很弱。这才是下一步最该提升的地方。
5. 延迟里包含了串行调用三个数据源的限速等待，没有并发和缓存。

## 单元测试

    48 passed

覆盖：三个源解析器、去重合并、分块、哈希嵌入、向量检索与持久化、评测指标、引用解析、prompt 组装、RAG 流程（离线 ScriptedClient）、裁判判定、API 端点（FastAPI TestClient + 假数据源 + 语料缓存）。全部离线可跑，不需要 API key。

## 状态

W1 完成：多源检索 + 去重归一化。
W2 完成：索引层 + 评测骨架 + 检索指标。
W3 完成：DeepSeek 客户端 + 带引用 RAG + 裁判负对照。
W4 进行中：FastAPI + 内置 Web UI 完成并实测通过；部署与（可选）Next.js 前端待续。
