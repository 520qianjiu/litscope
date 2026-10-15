# LitScope

输入一个研究主题，得到一份每个结论都能回溯到原文的工作草稿。

![架构](docs/assets/architecture.png)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/520qianjiu/litscope)

## 解决什么问题

做综述时，检索、去重、归纳、标注出处是四件分离的手工活。LitScope 把它们串成一条可评测的流水线：多源检索 -> 去重归一化 -> 分块嵌入 -> 检索 -> 带引用生成 -> 引用审计。

## 快速上手

    pip install -r requirements.txt
    python scripts/check_env.py          # 检查 .env
    python scripts/ping_llm.py           # 验证 LLM 连通
    python scripts/serve.py              # 起服务

浏览器打开 http://127.0.0.1:8000。

![真实运行](docs/assets/demo.gif)

上面这段是真实运行的输出回放（检索 + DeepSeek 生成），不是手画的示意图。素材由 scripts/gen_assets.py 生成。

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

### 2. 检索器对照：BM25 vs bge-m3（本项目最有信息量的一组数字）

同一语料（44 篇论文 / 205 chunks），两种查询模式。

#### A. 英文关键词查询（20 条）

| 检索器 | R@1 | R@3 | R@5 | R@10 | MRR | 延迟 P50 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25（词法） | 0.95 | 1.00 | 1.00 | 1.00 | 0.975 | 0.5 ms |
| bge-m3（1024 维） | 0.95 | 0.95 | 1.00 | 1.00 | 0.960 | 196.5 ms |

BM25 略胜。词法查询本来就是它的主场。

#### B. 中文查询（15 条，论文标题译成中文）

| 检索器 | recall@5 | MRR | 返回空结果 | 延迟 P50 |
| --- | --- | --- | --- | --- |
| BM25（词法） | 0.1333 | 0.1333 | 13 / 15 | 0.1 ms |
| bge-m3（1024 维） | 1.0000 | 0.9333 | 0 / 15 | 198.6 ms |

中文查询下 BM25 有 15 条里的 13 条完全检不到东西，bge-m3 一条不落。

#### C. 结论：按场景选，不是二选一

| 场景 | 用什么 | 理由 |
| --- | --- | --- |
| 英文查询 | BM25（默认） | 效果持平，快约 400 倍，零成本、零外部依赖 |
| 中文查询 | dense（bge-m3） | BM25 不可用，只有多语言嵌入能跨语言 |

复现：

    python eval/compare_retrievers.py --topic graph-neural-network --papers 15 --questions 20 --query-mode keywords --top-k 10
    python eval/compare_retrievers.py --topic graph-neural-network --papers 15 --questions 15 --query-mode zh --top-k 5 --out metrics_compare_zh.json

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

平衡准确率 0.5 意味着裁判是橡皮图章（永远说 SUPPORTED），1.0 意味着完美区分。所以第 3 节那个引用支持率不是白来的——裁判确实会拒绝。


### 5. 全文 vs 摘要：客观评测（n=8）

标签是**自动生成且客观的**：从论文正文（METHODS / RESULTS）随机取一句 70–320 字符的话作为标准答案，再测检索到的 chunk 里到底有没有这句话。这句话按构造不可能出现在摘要里。

| 方案 | evidence_recall | refusal_rate | citation_support |
| --- | --- | --- | --- |
| 仅摘要（dense） | 0.000 | 0.875 | 0.4167 |
| 两级全文检索 | 0.125 | 0.750 | 0.6667 |
| 两级全文 + 交叉编码器重排 | **0.375** | **0.625** | **0.8333** |

规模：80 个摘要 chunk vs 1625 个全文 chunk（20 倍）。语料 18 篇论文，其中 7 篇成功拿到全文。

**怎么读这组数字：**

1. **摘要组 8/8 全部拒答** —— 它诚实地说「资料不足以回答」，因为证据确实不在摘要里
2. **全文组拒答率降到 62.5%**，引用支持率从 0.375 提升到 0.750 —— 全文确实带来了可用的证据
3. **但 evidence_recall 只有 0.125** —— 检索很少正好命中那句原话

第 3 点最有价值：**瓶颈已经从「没有数据」变成「检索不准」** —— 于是加了第三组。

**重排（`BAAI/bge-reranker-v2-m3` 交叉编码器）把 evidence_recall 从 0.125 提到 0.375，三个指标同时改善。** 原因很直接：向量检索把 query 和段落**分开**编码，分不清「共享词汇」和「真的回答问题」；交叉编码器**一起读**，能把前者拒掉。

三个指标单调改善（0.000 → 0.125 → 0.375、0.875 → 0.750 → 0.625、0.4167 → 0.6667 → 0.8333），说明**每个组件都在贡献**，没有凑数的部分。

诚实说明：`evidence_recall` 用严格子串匹配（前 120 字符），chunk 边界切分、或返回了相关但不同的段落都会判为 miss，所以这个数字**偏保守**。

复现：

    python eval/fulltext_eval.py --topic graph-neural-network --papers 6 --questions 8 --out eval/metrics_fulltext.json

### 6. 冷启动优化：串行嵌入 -> 4 路并行

线上首次提问要现建索引（检索 -> 下载全文 -> 嵌入），其中嵌入是绝对大头，批次原本逐条串行发送。同一批 1429 个 chunk 实测（bge-m3）：

| 嵌入方式 | 1429 chunks 耗时 |
| --- | --- |
| 串行批次（旧默认） | 16.3 s |
| 4 路并发（新默认） | 4.1 s |
| 6 路并发 | 3.1 s |

端到端本地冷启动（检索 2.8 s + 下载 1.1 s + 嵌入 4.5 s + 生成 1.8 s）：**21.0 s -> 11.1 s**；同主题二次提问（内存缓存命中）2.9 s。

实现：`EMBEDDING_CONCURRENCY`（默认 4）控制嵌入批次并发，`LITSCOPE_FULLTEXT_CONCURRENCY`（默认 4）控制全文下载并发；两者都做容错解析，乱填不会让服务崩溃。`/api/ask` 响应新增 `timings` 字段（sources / abstract_index / fulltext_fetch / fulltext_index / answer），线上冷启动可逐段归因。

## 踩过的一个真坑：跨语言静默失败

最初版本用我自己写的 256 维哈希嵌入，并用 BM25 做词法基线。跑中文提问时出现了这个：

    问题：图神经网络靠什么机制传播信息？
    检索返回 5 个段落，全部 score=0.0
    模型照常回答，引用写成 [1][2][3]

那 5 个段落其实是脉冲神经网络和音乐识别的论文——因为中文 token 与英文 token 零重叠，BM25 全部得分 0，排序退化成按文档顺序取前 5。系统没有任何报警，模型也没拒答。

两步修复：

1. app/index/bm25.py 加 min_score 过滤：零重叠返回空，而不是返回垃圾。补了 4 个测试守住，包括 test_pipeline_reports_empty_instead_of_garbage（并断言此时不调用 LLM）。
2. 接入多语言嵌入 bge-m3，中文查询 recall@5 从 0.13 变成 1.00。

这个坑的价值在于它不报错：没有异常、没有 500、日志干净，只是答案悄悄变错。评测脚本是唯一能发现它的东西——这也是先搭评测再调优的理由。

## 部署踩坑记录：一个环境变量让服务起不来

部署到 Railway 后服务反复崩溃、域名返回 502，平台日志里只有一行和真实原因毫不相干的报错：

    File "/app/app/index/embedder.py", line 90, in build_embedder
        dim=int(os.environ.get("EMBEDDING_DIM") or 256)
    ValueError: invalid literal for int() with base 10: "BAAI/bge-m3"

### 根因

哈希嵌入器会读 EMBEDDING_DIM，而平台上这一项被误填成了模型名 BAAI/bge-m3（正确的归属是 EMBEDDING_MODEL）。代码直接 int() 硬转，**在模块导入期**抛异常，uvicorn 随即退出——整个服务打不开。

### 为什么这个失败模式很危险

1. **报错位置和真实错误无关**：看到的是「嵌入维度解析失败」，真正的问题是「变量填串了」。
2. **发生在导入期**：create_app() 在模块顶层执行，任何配置异常都会让进程直接死掉，而不是返回一个 500。
3. **我的第一次修复是错的**：我加了一层 try/except，在异常分支里回退调用 uild_embedder(hashing)——而那个调用**同样会读 EMBEDDING_DIM**，于是回退也炸了。安全网自己踩了同一个坑。

### 修复

1. _as_int() 容错解析：非法数值回退默认值，绝不抛异常。
2. 回退路径再加一层兜底：直接 HashingEmbedder() 用硬编码默认值构造。
3. **失败要可见而非致命**：配置错误不再让服务崩溃，而是通过 /api/health 报告——embedder_error 字段给出原因。
4. ?debug=1 时列出非密钥变量的**实际值**，便于从外部定位。这次就是靠它一眼看出 EMBEDDING_PROVIDER=hashing、LITSCOPE_RETRIEVER=bm25 与预期不符。

### 教训

配置错误应该**降级运行并大声报告**，而不是让进程静默死掉——「服务根本打不开」是最难排查的失败模式：没有页面、没有接口，只能翻平台日志。

## 已知限制

- 全文只覆盖 arXiv 与 PMC 开放获取子集，其余记录仍是摘要级
- 评测标签偏简单（标题译中文 / 关键词查询）；英文关键词模式下 Recall@5 已饱和，区分度主要来自 R@1 与 MRR
- 裁判与生成同源（deepseek-flash），负对照只做了部分缓解，不能完全排除自我偏好
- dense 检索每次查询要一次 embedding API 调用（约 200 ms），未做查询向量缓存
- 多源检索（三个源的查询）仍是串行；全文下载与嵌入批次为有限并发（默认 4 路）；语料索引只存在内存里，进程重启后需重建

## 架构

五层，每层可独立测试：

    数据层  app/sources/    arXiv / Crossref / PubMed 适配器，统一 Paper
    处理层  app/dedupe.py   DOI -> 标题规范化 -> 模糊匹配 0.92
            app/index/      分块 + 嵌入（可插拔：哈希 / OpenAI 兼容）
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

语料按 topic 缓存在内存中，同主题二次提问不再重建索引。响应里的 timings 字段（sources / abstract_index / fulltext_fetch / fulltext_index / answer）可逐段核对耗时；线上演示前先把要问的问题问一遍即可预热。

## 配置

    LLM_PROVIDER=deepseek
    DEEPSEEK_API_KEY=...

跨语言检索需要多语言嵌入，填这三项后把 retriever 切到 dense：

    EMBEDDING_PROVIDER=openai
    EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
    EMBEDDING_API_KEY=...
    EMBEDDING_MODEL=BAAI/bge-m3

冷启动调优（可选，代码里都有默认值）：

    EMBEDDING_CONCURRENCY=4           # 嵌入批次并发数，默认 4
    LITSCOPE_FULLTEXT_CONCURRENCY=4   # 全文下载并发数，默认 4

## 部署

见 docs/DEPLOY.md（Render / Railway 单服务部署，含上线自检清单）。

## 单元测试

    python -m unittest discover -s tests   # 129 passed

全部离线可跑，不需要 API key（靠 ScriptedClient + 假数据源）。覆盖：三个源解析器、去重合并、分块、哈希嵌入、向量检索与持久化、BM25、零分词重叠过滤、引用解析、prompt 组装、RAG 流程、裁判判定、API 端点与语料缓存、嵌入/下载并发与分阶段计时。

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
- W5 BM25 基线 + 跨语言静默失败修复 + bge-m3 对照评测
- W6 上线部署（Railway）+ 配置容错改造 + 部署踩坑复盘
- W7 arXiv / PMC 全文检索（章节感知分块 + 两级检索）+ 客观评测
- W8 交叉编码器重排（三个指标同时提升）
- W9 冷启动优化（并行嵌入/下载 + 分阶段计时）

下一步：线上复测冷启动数字并准备演示。
