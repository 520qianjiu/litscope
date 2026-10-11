# LitScope 设计文档 v0.1

日期：2026-10-12
状态：待评审

## 1. 要解决的问题

做文献综述时，研究者要跨多个来源检索、去重、通读、归纳，而且每个结论最后都得能回溯到原文。

现有工具要么只做检索（PubMed / Google Scholar），要么只做问答（通用 LLM 聊天）。中间这一段——把检索结果变成带引用、可核对的知识——全靠手工。

LitScope 的目标：输入一个研究主题，输出一份每个结论都带原文出处的工作草稿，并且这件事本身是可评测的。

## 2. 三条硬要求

1. 真实数据、真实用途：arXiv / Crossref / PubMed 公开 API，全部免 key；产出能直接用于写相关工作章节。
2. 可在线访问：后端 Railway，前端 Vercel。
3. README 写明：解决什么问题、架构、量化结果（见第 6 节）。

## 3. 架构

分层，每层可独立测试：

    检索层   app/sources/   多源适配器，统一输出 Paper
    归一化   app/dedupe.py  去重与字段合并
    索引层   app/index/     分块 + embedding + 向量库
    生成层   app/rag/       检索 -> 组装上下文 -> LLM -> 带引用回答
    评测层   eval/          评测集 + 指标脚本
    服务层   app/api/       FastAPI
    界面层   web/           Next.js

数据流：

    主题 -> 检索层 -> Paper[] -> 归一化 -> Paper[]
         -> 索引层 -> vectors
         -> 问题 -> 向量检索 -> top-k chunks -> 生成层 -> 答案 + 引用
         -> 评测层 -> metrics.json

## 4. 数据模型

Paper：source, source_id, title, abstract, authors, year, venue, doi, url, citations, merged_sources

Chunk：paper_id, chunk_index, text, embedding

Citation：paper_id, chunk_index, span（答案中被支撑的那句话）

## 5. 接口（第一版）

    GET  /api/health
    GET  /api/search?q=&limit=&sources=        检索并去重，返回 Paper[]
    POST /api/index   {papers: []}             建索引，返回 chunk 数
    POST /api/ask     {question, top_k}        返回 {answer, citations[]}

## 6. 量化指标（README 必须出现）

- 检索 Recall@5 / Recall@10：自建 30-50 条评测集，人工标注相关论文
- 引用准确率：答案里的每条引用是否真的支持对应结论；LLM-as-judge 先跑，再人工抽查 30 条校准
- 端到端延迟 P50 / P95
- 单次问答成本：按 token 估算，写清算法

## 7. 技术选型

| 层 | 选型 | 理由 |
| --- | --- | --- |
| HTTP | httpx | 支持异步，与 FastAPI 同栈 |
| 向量库 | Chroma（本地）到 pgvector（部署） | 先快后稳 |
| LLM | 可插拔，默认 DeepSeek | 便宜，国内可直连 |
| 后端 | FastAPI + uvicorn | 招聘常见栈 |
| 前端 | Next.js | 部署 Vercel，作品观感好 |
| 评测 | 自写脚本 + 评测集 JSON | 零额外依赖 |

## 8. 里程碑

- W1 检索层 + 归一化，命令行可用（本周）
- W2 索引层 + 评测集 + 第一组数字
- W3 生成层 + FastAPI + Next.js
- W4 部署上线 + README（架构图、指标、demo、复现步骤）

## 9. 范围边界（不做的事）

- 不做全文 PDF 下载与解析：第一版只用摘要与元数据
- 不做多用户、鉴权、计费
- 不做查重，不做代写
- 不做自动投稿

## 10. 风险

- 摘要信息量有限，引用准确率可能偏低 -> 评测集必须包含人工核对环节
- 公开 API 有速率限制 -> 加本地缓存与指数退避
- 部署在境外，国内直连慢 -> README 提供本地运行方式作为备选

## 11. 验收标准

1. 命令行能对真实主题检索三个来源并去重
2. 评测脚本能输出 Recall 与引用准确率两个数字
3. 线上地址能回答一个问题并给出可核对的引用
4. README 能让陌生人 10 分钟内复现


## 12. 变更记录

2026-10-12（W2 完成时）：

- 向量库：原定 Chroma，改为**先实现纯 Python 内存向量库**（app/index/store.py），支持 JSON 持久化。
  原因：开发环境无 numpy，且第一版规模（数千 chunk）纯 Python 足够；Chroma / pgvector 留到部署阶段做后端替换。
- Embedding：**DeepSeek 不提供 embeddings 接口**（它只有对话模型）。
  因此 app/index/embedder.py 做成可插拔：
  - HashingEmbedder：离线确定性哈希嵌入，零依赖，用于离线测试与无 key 场景
  - OpenAICompatEmbedder：任何 OpenAI 兼容的 embeddings 端点（bge-m3 / 智谱 / OpenAI）
  默认 provider = hashing；W3 视评测结果决定是否切换到真实嵌入模型。
- 评测：新增 **title-recovery** 标签方案（自动生成、无需人工标注）作为 W2 的客观指标；
  原定的语义相关性评测推到 W3（需要 LLM key 做 judge）。
- 指标口径：percentile 采用线性插值（与 numpy.percentile 一致）。


2026-10-12（W3 完成时）：

- 新增 LLM 层 app/llm.py：OpenAI 兼容客户端（默认 DeepSeek）+ 离线 ScriptedClient（单测用无 key）。
- 新增 RAG 层 app/rag/：prompt 组装、引用解析（含无效标记检测）、检索问答流水线。
- 新增裁判层 eval/judge.py：citation support 判定；注意 UNSUPPORTED 是 SUPPORTED 的超串，判定顺序必须先判 UNSUPPORTED。
- 新增负对照 eval/judge_sanity.py：把 claim 配到别的论文的段落上，用平衡准确率衡量裁判是否只是橡皮图章。
- 指标口径修正：原 judge_discrimination = 正例率 - 负例拒绝率 是错的（正负都满分时算出 0），改为 balanced accuracy = (正例率 + 负例拒绝率) / 2。
