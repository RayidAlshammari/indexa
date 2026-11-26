---
title: Indexa
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Government index readiness assessment (NDI, Arabic)
---

# Indexa | مستشار جاهزية المؤشرات الحكومية

Demo of the Indexa assessment engine. `app.py` downloads the code from GitHub at startup; `Dockerfile` is the alternative for paid Docker Spaces. Source: https://github.com/RayidAlshammari/indexa

Space secrets and variables (Settings → Variables and secrets):

| Name | Kind | Value |
|---|---|---|
| `INDEXA_LLM_API_KEY` | Secret | Groq API key |
| `INDEXA_LLM_BASE_URL` | Variable | `https://api.groq.com/openai/v1` |
| `INDEXA_LLM_MODEL` | Variable | `qwen/qwen3.8-27b` |
| `INDEXA_DEMO_TOKEN` | Secret (optional) | token required to start sample runs |
