# 部署指南（单服务：FastAPI + 内置 UI）

目标：把 LitScope 部署成一个可公开访问的网址，满足「能在线访问」这条硬要求。

## 0. 前置：装 Git

这台机器上没有 git，先装：

    https://git-scm.com/download/win

装完关掉并重开终端，然后确认：

    git --version

## 1. 本地先跑通

    cd D:\CodexHome\2026-10-08\nature-skills\outputs\litscope
    pip install -r requirements.txt
    python scripts/ping_llm.py
    python scripts/serve.py

浏览器打开 http://127.0.0.1:8000 应该能看到界面。

## 2. 建仓库并推上 GitHub

    git init
    git add .
    git commit -m 'LitScope: multi-source literature RAG with citation auditing'
    git branch -M main
    git remote add origin https://github.com/<你的用户名>/litscope.git
    git push -u origin main

推送前务必确认 .env 没被提交。检查已跟踪文件清单：

    git ls-files

清单里只应该出现 .env.example，不能出现 .env。如果出现了，立刻停：

    git rm --cached .env

然后去 DeepSeek 平台把这个 key 删掉重建。

## 3. 部署（二选一）

### Render（推荐，配置最省事）

1. 用 GitHub 登录 https://render.com
2. New -> Web Service -> 选 litscope 仓库
3. Render 会自动读取仓库里的 render.yaml
4. 在 Environment 里手动填 DEEPSEEK_API_KEY（render.yaml 里标了 sync: false）
5. Deploy

### Railway

    Build:  pip install -r requirements.txt
    Start:  python scripts/serve.py
    Health: /api/health

记得在 Variables 里加 DEEPSEEK_API_KEY。

## 4. 上线后自检

    https://<服务名>.onrender.com/api/health

返回 status ok 且 llm ready 就算成功。然后逐项确认：

1. 首页能打开，输入主题和问题能出带引用的回答
2. 点开引用条目能核对到具体 chunk 文本
3. 同一主题再问一次响应明显变快（语料缓存生效）
4. 故意问一个语料里没有的问题，确认模型拒答而不是编造

## 5. 免费档的两个坑

- Render 免费实例闲置会休眠，首次访问要等约 30 秒唤醒。面试演示前先自己访问一次预热。
- 免费档内存 512MB，语料别一次拉太多。当前默认每源 8 篇、去重后约 20 篇，完全够用。
