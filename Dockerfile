FROM python:3.11-slim

LABEL maintainer="智能投顾AI助手系统"
LABEL description="基于LangChain+LangGraph的智能投顾AI助手系统"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY web_api.py .
COPY main.py .
COPY db.py .
COPY auth.py .
COPY llm_factory.py .
COPY .env.example .env

# 复制三个子系统
COPY "01-私募基金运作指引问答助手（反应式）" "./01-私募基金运作指引问答助手（反应式）"
COPY "02-智能投研助手（深思熟虑）" "./02-智能投研助手（深思熟虑）"
COPY "03-投顾AI助手（混合式）" "./03-投顾AI助手（混合式）"

# 创建日志目录
RUN mkdir -p logs

# 暴露端口
EXPOSE 8002

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/')" || exit 1

# 启动命令
CMD ["python", "web_api.py"]
