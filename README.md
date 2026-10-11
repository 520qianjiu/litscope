# LitScope

输入一个研究主题，得到一份每个结论都能回溯到原文的工作草稿。

![架构](docs/assets/architecture.png)

## 解决什么问题

做综述时，检索、去重、归纳、标注出处是四件分离的手工活。LitScope 把它们串成一条可评测的流水线：多源检索 -> 去重归一化 -> 分块嵌入 -> 检索 -> 带引用生成 -> 引用审计。

## 快速上手

    pip install -r requirements.txt
    python scripts/check_env.py          # 检查 .env
    python scripts/ping_llm.py           # 验证 LLM 连通
    python scripts/serve.py              # 起服务

浏览器打开 http://127.0.0.1:8000。

![真实运行](docs/assets/demo.gif)

上面这段是真实运行的输出回放（BM25 检索 + DeepSeek 生成），不是手画的示意图。素材由 scripts/gen_assets.py 生成。

## 量化结果

所有数字都可用仓库里的脚本复现，原始结果在 metrics*.json。

### 1. 检索阶段（多源召回 + 去重）

| 指标 | 值 |
| --- | --- |
| Recall@1 | 0.9167 |
| Recall@5 | 1.0000 |
| Recall@10 | 1.0000 |
| MRR | 0.9583 |
| 延迟 P50 / P95 | 7.06 s / 10.30 s |
| 评测条数 | 12（失败 0） |

### 2. 检索器对照：BM25 vs 哈希嵌入

同一语料（44 篇论文 / 205 chunks）、同一组 20 条关键词查询：

| 检索器 | R@1 | R@3 | R@5 | R@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| BM25（词法） | 0.95 | 1.00 | 1.00 | 1.00 | 0.975 |
| 哈希嵌入（256 维） | 0.90 | 1.00 | 1.00 | 1.00 | 0.942 |

**结论：30 行纯 Python 的 BM25 打赢了我手写的哈希嵌入。** 所以 BM25 现在是默认检索器。

这不算丢人，反而是基准测试该起的作用——不测就不知道自己在优化什么。原计划直接上向量检索，测完改成先把词法基线做扎实。

复现：

    python eval/compare_retrievers.py --topic graph-neural-network --papers 15 --questions 20 --query-mode keywords --top-k 10

### 3. RAG 端到端（6 个问题，top_k=5）

| 指标 | 值 |
| --- | --- |
| 目标论文检索率 | 1.0000 |
| 引用支持率（LLM 裁判） | 1.0000（12 对） |
| 无效引用标记数 | 0 |
| 延迟 P50 / P95 | 1.62 s / 1.99 s |
| token 总数 / 均值 | 3922 / 653.7 |

### 4. 裁判负对照

让同一个 claim 去配另一篇论文的段落，看裁判会不会照单全收：

| 指标 | 值 |
| --- | --- |
| 正例支持率（claim + 自己的来源） | 1.0000（6/6） |
| 负例拒绝率（claim + 别人的来源） | 1.0000（6/6） |
| 平衡准确率 | 1.0000 |

平衡准确率 0.5 意味着裁判是橡皮图章（永远说 SUPPORTED），1.0 意味着完美区分。所以第 3 节那个 1.0000 的引用支持率不是白来的——裁判确实会拒绝。

## 已知限制

### 中文问题检索不到英文语料（重要）

    python scripts/ask.py --topic graph-neural-network --retriever bm25 --question 图神经网络靠什么机制传播信息

返回空结果并明确说明原因。BM25 是词面匹配，中文 token 与英文 token 零重叠，所有得分都是 0；哈希嵌入是字符级哈希，同样没有跨语言能力。

我是在实测里撞到这个才发现的：最初版本会把这些 0 分结果照常喂给模型，模型于是拿脉冲神经网络、音乐识别的段落去回答图神经网络的问题，还一本正经地给出引用。**修法是让检索在无重叠时返回空，而不是返回垃圾**（app/index/bm25.py 的 min_score 过滤），并补了 4 个测试守住这个行为。

根治方向：换多语言嵌入模型（bge-m3 / 智谱 embedding-3，中英双语），走 OpenAICompatEmbedder，填三个环境变量即可切换。这是下一优先级——不是优化，是功能补齐。

其他限制：

- 只吃摘要与元数据，不解析全文 PDF
- 评测标签偏简单（标题 / 关键词查询），Recall@5 已饱和，说明不了语义检索强弱
- 裁判与生成同源（deepseek-flash），负对照只做了部分缓解
- 延迟含串行调用三个数据源的限速等待，未做并发与缓存

## 架构

五层，每层可独立测试：

    数据层  app/sources/    arXiv / Crossref / PubMed 适配器，统一 Paper
    处理层  app/dedupe.py   DOI -> 标题规范化 -> 模糊匹配 0.92
            app/index/      分块 + 嵌入（可插拔）
    索引层  app/index/      向量库（纯 Python 余弦）与 BM25
    生成层  app/rag/        强制引用的 prompt -> DeepSeek -> 引用回指
    评测层  eval/           R@k / MRR / 引用支持率 / 裁判负对照

## Web 服务

    python scripts/serve.py

| 接口 | 说明 |
| --- | --- |
| GET / | 内置 Web UI |
| GET /api/health | 嵌入器、检索器、LLM 状态 |
| GET /api/search?q=&limit=&sources= | 多源检索 + 去重 |
| POST /api/ask | {topic, question, top_k, per_source, retriever} |

语料按 topic 缓存，同主题二次提问不重复检索。

## 部署

见 docs/DEPLOY.md（Render / Railway 单服务部署，含上线自检清单）。

## 单元测试

    67 passed

全部离线可跑，不需要 API key（靠 ScriptedClient + 假数据源）。覆盖：三个源解析器、去重合并、分块、哈希嵌入、向量检索与持久化、BM25、零分词重叠过滤、引用解析、prompt 组装、RAG 流程、裁判判定、API 端点与语料缓存。

## 项目结构

    app/        核心代码（sources / index / rag / api / llm）
    eval/       评测脚本与指标
    scripts/    CLI 入口与资产生成
    tests/      单元测试
    docs/       设计文档、部署指南、图片素材
    metrics*.json  评测原始结果

## 状态

- W1 多源检索 + 去重归一化
- W2 索引层 + 评测骨架 + 检索指标
- W3 DeepSeek 客户端 + 带引用 RAG + 裁判负对照
- W4 FastAPI + 内置 Web UI + 部署配置 + README 素材

下一步：多语言嵌入（解决中文查询）+ 线上部署。
