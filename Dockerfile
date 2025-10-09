FROM python:3.10-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式檔案
COPY . .

# 創建必要的目錄
RUN mkdir -p downloaded_videos templates static

# 暴露端口
EXPOSE 7860

# 啟動應用
CMD ["python", "app.py"]
