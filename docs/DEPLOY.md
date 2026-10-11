# 部署指南（单服务：FastAPI + 内置 UI）

目标：把 LitScope 部署成一个可公开访问的网址，满足「能在线访问」这条硬要求。

## 0. 前置

- Git：已装（D:\\Git）。若终端提示找不到 git，重开一个终端。
- DeepSeek API key：https://platform.deepseek.com/api_keys
- 硅基流动 API key（多语言嵌入）：https://siliconflow.cn -> API 密钥

## 1. 本地先跑通

    cd D:\\projects\\litscope
    pip install -r requirements.txt
    python scripts/check_env.py
    python scripts/ping_llm.py
    python scripts/serve.py

浏览器打开 http://127.0.0.1:8000，用中文提一个问题，确认能出带引用的回答。

## 2. 先在 GitHub 建空仓库

打开 https://github.com/new

    Repository name: litscope
    Visibility:      Public 或 Private 都行
    Add a README file:        不勾
    Add .gitignore:           不选
    Choose a license:         不选

必须都是空的。否则 push 会因为历史不一致被拒。

## 3. 推送

    cd D:\\projects\\litscope
    git remote add origin https://github.com/<你的用户名>/litscope.git
    git push -u origin main

推送前先确认 .env 没有被跟踪：

    git ls-files

输出里只能有 .env.example，绝不能有 .env。本仓库已把 .env 写进 .gitignore 并验证过，但这一步值得每次做。

## 4. 部署到 Render

1. 用 GitHub 账号登录 https://render.com
2. New -> Web Service -> 选择 litscope 仓库
3. Render 会自动读取仓库里的 render.yaml（build / start / health check 都已配好）
4. 在 Environment 里**手动填两个密钥**（其余已在 render.yaml 里声明）：

| Key | 填什么 |
| --- | --- |
| DEEPSEEK_API_KEY | 你的 DeepSeek key |
| EMBEDDING_API_KEY | 你的硅基流动 key |

render.yaml 已经声明好的（不用重复填）：

| Key | Value |
| --- | --- |
| PYTHON_VERSION | 3.12.7 |
| LLM_PROVIDER | deepseek |
| DEEPSEEK_BASE_URL | https://api.deepseek.com |
| DEEPSEEK_MODEL | deepseek-chat |
| LITSCOPE_RETRIEVER | dense |
| EMBEDDING_PROVIDER | openai |
| EMBEDDING_BASE_URL | https://api.siliconflow.cn/v1 |
| EMBEDDING_MODEL | BAAI/bge-m3 |

注意 LITSCOPE_RETRIEVER=dense 是刻意设的：线上演示面向中文提问，BM25 对中文查询会返回空（见 README 的跨语言那一节）。

5. 点 Deploy，等两分钟。

## 5. 上线后自检

    https://<服务名>.onrender.com/api/health

期望返回 status ok、embedder openai-compatible、retriever dense、llm ready。然后逐项确认：

1. 首页打开，用**中文**提问，能出带引用的回答
2. 点开引用条目，能核对到具体 chunk 文本
3. 同一主题再问一次，响应明显变快（语料缓存生效）
4. 故意问一个语料里没有的问题，确认模型拒答而不是编造
5. 把 /api/health 的 retriever 与本地对比，确认线上确实用的是 dense

## 6. 免费档的两个坑

- Render 免费实例闲置会休眠，首次访问要等约 30 秒唤醒。面试演示前先自己访问一次预热。
- 免费档内存 512MB，语料别一次拉太多；当前默认每源 8 篇、去重后约 20 篇，够用。
- dense 检索每次查询会调一次 embedding API（约 200 ms），免费额度足够演示，但别拿它压测。
