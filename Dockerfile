FROM python:3.10-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# 創建非 root 用戶
RUN useradd --create-home --shell /bin/bash app

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式檔案
COPY . .

# 創建必要的目錄並設定權限
RUN mkdir -p /tmp/downloaded_videos && \
    touch /tmp/downloaded_videos.json /tmp/processed_count.json && \
    chown -R app:app /tmp && \
    chmod -R 755 /tmp

# 切換到非 root 用戶
USER app

# 確保應用用戶可以寫入 /tmp
RUN touch /tmp/downloaded_videos.json /tmp/processed_count.json

# 暴露端口
EXPOSE 7860

# 啟動應用
CMD ["python", "app.py"]
