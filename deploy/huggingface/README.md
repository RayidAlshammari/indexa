---
title: Indexa
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Government index readiness assessment (NDI, Arabic)
---

# Indexa | مستشار جاهزية المؤشرات الحكومية

Demo of the Indexa assessment engine. Source: https://github.com/RayidAlshammari/indexa

Space secrets and variables (Settings → Variables and secrets):

| Name | Kind | Value |
|---|---|---|
| `INDEXA_LLM_API_KEY` | Secret | Groq API key |
| `INDEXA_LLM_BASE_URL` | Variable | `https://api.groq.com/openai/v1` |
| `INDEXA_LLM_MODEL` | Variable | `qwen/qwen3.8-27b` |
| `INDEXA_DEMO_TOKEN` | Secret (optional) | token required to start sample runs |
