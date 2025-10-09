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

# 創建資料目錄並設定權限
RUN mkdir -p /app/data && \
    chmod 777 /app/data

# 設定環境變數指向可寫入的目錄
ENV DATA_DIR=/app/data
# 強制 Python 無緩衝輸出，確保日誌即時顯示
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 7860

# 啟動應用（使用 -u 參數確保無緩衝輸出）
CMD ["python", "-u", "app.py"]
