FROM python:3.11-slim-bookworm

WORKDIR /app

# 安裝系統依賴（包含 OpenCV 和 MediaPipe 所需的庫）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY /config/requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式檔案
COPY . .

# 創建資料目錄並設定權限
RUN mkdir -p /app/data /app/data/downloaded_videos && \
    chmod -R 777 /app/data

# 設定環境變數指向可寫入的目錄
ENV DATA_DIR=/app/data
# 強制 Python 無緩衝輸出，確保日誌即時顯示
ENV PYTHONUNBUFFERED=1
# Keras backend
ENV KERAS_BACKEND=tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=2

# 🚫 禁用所有 GPU/Metal/OpenGL 加速（強制 CPU-only）
ENV CUDA_VISIBLE_DEVICES=-1
ENV MEDIAPIPE_GPU_DISABLED=1
ENV MEDIAPIPE_DISABLE_GPU=1
ENV GLOG_logtostderr=1
# 禁用 EGL（OpenGL 的 headless 渲染）
ENV MEDIAPIPE_DISABLE_EGL=1
ENV EGL_PLATFORM=surfaceless
# 抑制 MediaPipe GPU 試探的錯誤訊息（0=INFO, 1=WARNING, 2=ERROR, 3=FATAL）
ENV GLOG_minloglevel=2

# Render 會動態設定 PORT，預設 10000
ENV PORT=10000

# 暴露端口（Render 會自動映射）
EXPOSE 10000

# 啟動應用（使用 -u 參數確保無緩衝輸出）
CMD ["python", "-u", "app.py"]
