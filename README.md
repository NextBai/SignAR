# 手語影片識別系統 🏭🤟

基於 Flask + WebSocket 的手語影片識別系統，整合 Facebook Messenger Bot 和深度學習模型，提供即時手語識別與翻譯服務。

## 功能特點

- 🤟 **手語識別**：使用 BiGRU + 注意力機制識別手語動作
- 🎨 **生產線視覺介面**：動畫化的影片處理流程
- 📹 **Messenger Bot 整合**：接收 Facebook Messenger 傳來的影片
- 🎬 **即時影片播放**：前端自動載入並播放收到的影片
- 🤖 **AI 句子重組**：使用 OpenAI 將識別結果組合成完整句子
- 💾 **智慧重複檢測**：MD5 哈希值避免重複處理
- 💬 **WebSocket 即時同步**：Messenger Bot 觸發前端動畫
- 📊 **統計追蹤**：即時處理數量統計

## 系統架構

### 技術堆疊
- **前端**: HTML/CSS/JavaScript + Socket.IO
- **後端**: Flask + Flask-SocketIO + Eventlet
- **深度學習**: 
  - TensorFlow/Keras (模型推論)
  - PyTorch (特徵提取)
  - MediaPipe (骨架檢測)
  - OpenCV (影片處理)
- **AI 服務**: OpenAI GPT (句子重組)
- **部署**: Render / Docker
- **影片儲存**: 本地檔案系統（需持久化儲存）

### 識別流程
1. **影片接收**：從 Messenger 接收手語影片
2. **特徵提取**：
   - RGB 特徵（ResNet-50）
   - 骨架特徵（MediaPipe Holistic）
3. **滑動窗口識別**：80 幀/窗口，識別每個手語單詞
4. **句子重組**：使用 OpenAI 將單詞組合成完整句子
5. **結果回傳**：透過 Messenger 發送識別結果

## 使用方式

### Messenger Bot
1. 發送手語影片到 Facebook Messenger Bot
2. 系統自動下載並處理影片
3. 前端即時顯示生產線動畫
4. Bot 自動回覆識別的句子（例如："I love you"）

### 手語單詞支援
系統目前支援 15 個常用手語單詞：
- hello, thank you, please, sorry, love
- yes, no, help, student, teacher
- family, friend, name, how, what

## 環境變數設定

需要設定以下環境變數：

```bash
# Messenger Bot 設定（必需）
MESSENGER_VERIFY_TOKEN=你的驗證令牌
PAGE_ACCESS_TOKEN=你的頁面存取令牌

# OpenAI 設定（選填，用於句子重組）
OPENAI_API_KEY=你的 OpenAI API 金鑰

# 系統設定（選填）
SECRET_KEY=flask_secret_key
DATA_DIR=/app/data          # 影片儲存目錄
PORT=10000                  # 服務端口

# TensorFlow 設定（自動配置）
KERAS_BACKEND=tensorflow
TF_CPP_MIN_LOG_LEVEL=2
PYTHONUNBUFFERED=1
```

## 部署到 Render

### 前置準備
1. 確保專案包含以下檔案：
   - `app.py` - 主應用程式
   - `sliding_window_inference.py` - 手語識別核心
   - `requirements.txt` - Python 依賴
   - `Dockerfile` - Docker 配置
   - `model_output/` - 模型文件目錄
     - `best_model_mps.keras` - 訓練好的模型
     - `label_map.json` - 標籤映射
   - `feature_extraction/` - 特徵提取模組
   - `templates/` - 前端模板

### 1. 建立 Web Service
1. 登入 [Render](https://render.com/)
2. 點擊 "New +" → "Web Service"
3. 連接你的 GitHub 倉庫
4. 選擇此專案

### 2. 設定部署配置
- **Name**: `sign-language-recognition`（自訂）
- **Environment**: `Docker`（推薦）或 `Python 3`
- **Branch**: `main`

#### 使用 Docker 部署（推薦）
- **Build Command**: 自動偵測 Dockerfile
- **Start Command**: 自動使用 Dockerfile 的 CMD

#### 使用 Python 部署
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -u app.py`

### 3. 設定環境變數
在 Render Dashboard > Environment 中新增：

```bash
# 必需變數
MESSENGER_VERIFY_TOKEN=你的驗證令牌
PAGE_ACCESS_TOKEN=你的頁面存取令牌

# 選填變數
OPENAI_API_KEY=你的 OpenAI API 金鑰
DATA_DIR=/opt/render/project/data
PORT=10000

# 系統變數（自動配置）
KERAS_BACKEND=tensorflow
TF_CPP_MIN_LOG_LEVEL=2
PYTHONUNBUFFERED=1
```

### 4. 啟用持久化儲存（重要！）
手語影片需要保存在持久化儲存中：

- 在 Render Dashboard > Disks 新增 Disk
- **Name**: `video-storage`
- **Mount Path**: `/opt/render/project/data`（Python 部署）或 `/app/data`（Docker 部署）
- **Size**: 建議至少 2GB（每個影片約 5-50MB）

### 5. 資源配置建議
由於深度學習模型需要較多資源：

- **Instance Type**: Standard 或以上（建議 Standard Plus）
- **RAM**: 至少 2GB（模型載入需要約 500MB-1GB）
- **CPU**: 2 核心以上（推論速度更快）

### 6. 部署
點擊 "Create Web Service"，Render 會自動：
1. Clone 倉庫
2. 安裝依賴（可能需要 10-15 分鐘，包含 TensorFlow、PyTorch、MediaPipe）
3. 載入模型
4. 啟動服務

### 7. 驗證部署
部署完成後，訪問以下端點確認：

- `https://your-app.onrender.com/` - 前端介面
- `https://your-app.onrender.com/health` - 健康檢查
- `https://your-app.onrender.com/debug` - 系統資訊

健康檢查應返回：
```json
{
  "status": "healthy",
  "downloaded_videos_count": 0,
  "processed_count": 0,
  "verify_token_set": true,
  "page_token_set": true
}
```

## Messenger Webhook 設定

### 設定步驟
1. 前往 [Facebook Developers Console](https://developers.facebook.com/)
2. 選擇你的應用 → Messenger → 設定
3. 設定 Webhook URL：`https://your-app-name.onrender.com/webhook`
4. 設定 Verify Token 為你的 `MESSENGER_VERIFY_TOKEN`
5. 訂閱以下事件：
   - ✅ `messages`
   - ✅ `messaging_postbacks`
6. 選擇 Facebook 粉絲專頁並訂閱

### 測試 Webhook
```bash
# 測試驗證端點
curl "https://your-app.onrender.com/webhook?hub.mode=subscribe&hub.verify_token=YOUR_TOKEN&hub.challenge=test123"

# 應該返回：test123
```

### 常見問題

#### Q: Webhook 驗證失敗？
- 確認 `MESSENGER_VERIFY_TOKEN` 與 Facebook 設定一致
- 檢查 Render 服務是否正常運行
- 查看 Render Logs 確認收到驗證請求

#### Q: Bot 沒有回應？
- 確認 `PAGE_ACCESS_TOKEN` 正確且未過期
- 檢查粉絲專頁是否正確訂閱
- 查看 Render Logs 確認收到訊息

## Docker 部署（可選）

如果使用 Docker 部署到其他平台：

```bash
# 建立映像
docker build -t sign-language-recognition .

# 運行容器
docker run -d \
  -p 10000:10000 \
  -e MESSENGER_VERIFY_TOKEN=你的令牌 \
  -e PAGE_ACCESS_TOKEN=你的令牌 \
  -e OPENAI_API_KEY=你的金鑰 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/model_output:/app/model_output \
  sign-language-recognition
```

### Docker Compose（推薦）

建立 `docker-compose.yml`：

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "10000:10000"
    environment:
      - MESSENGER_VERIFY_TOKEN=${MESSENGER_VERIFY_TOKEN}
      - PAGE_ACCESS_TOKEN=${PAGE_ACCESS_TOKEN}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATA_DIR=/app/data
      - PORT=10000
    volumes:
      - ./data:/app/data
      - ./model_output:/app/model_output
    restart: unless-stopped
```

啟動：
```bash
docker-compose up -d
```

## API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 前端介面 |
| `/webhook` | GET | Webhook 驗證 |
| `/webhook` | POST | 處理 Messenger 訊息 |
| `/videos/<hash>` | GET | 取得影片檔案 |
| `/api/videos` | GET | 取得影片清單（JSON）|
| `/stats` | GET | 統計資訊 |
| `/health` | GET | 健康檢查 |
| `/debug` | GET | 調試資訊（包含模型狀態）|

## 專案結構

```
MVP拷貝2/
├── app.py                          # 主應用程式（Flask + 手語識別整合）
├── sliding_window_inference.py     # 滑動窗口手語識別核心
├── requirements.txt                # Python 依賴
├── Dockerfile                      # Docker 配置
├── README.md                       # 專案文檔
│
├── model_output/                   # 模型文件（需包含在部署中）
│   ├── best_model_mps.keras       # 訓練好的模型
│   └── label_map.json             # 標籤映射（15 個手語單詞）
│
├── feature_extraction/             # 特徵提取模組
│   ├── rgb_feature_extraction.py  # RGB 特徵（ResNet-50）
│   └── skeleton_feature_extraction.py  # 骨架特徵（MediaPipe）
│
├── templates/                      # 前端模板
│   └── index.html                 # 主頁面
│
├── config/                         # 配置文件
│   ├── requirements_inference.txt
│   └── .env
│
├── data/                          # 數據目錄（需持久化）
│   └── downloaded_videos/         # 儲存影片
│
├── scripts/                       # 其他腳本
└── training/                      # 訓練相關
```

## 支援的影片格式

MP4, AVI, MOV, MKV, FLV, WMV, WebM

## 影片處理規格

- **輸入**: 任意長度影片（最短 80 幀）
- **預處理**: 重採樣到 30 fps，resize 到 224x224
- **窗口大小**: 80 幀（約 2.67 秒）
- **特徵維度**: 300 幀 × 1119 維特徵
  - RGB: 512 維（ResNet-50）
  - 骨架: 607 維（MediaPipe Holistic）

## 模型資訊

- **架構**: BiGRU + 多頭注意力
- **訓練集**: 15 個手語單詞，每個約 88 幀
- **準確率**: 92-96%（v3.0 深度改進版）
- **推論速度**: 約 2-3 秒/窗口（CPU）
- **信心度**: 70-85%（經過校準）

## 影片管理

### 自動保留策略
- 所有透過 Messenger Bot 收到的影片會**保留**在伺服器上
- 影片存放於 `DATA_DIR/downloaded_videos/` 目錄
- 重複影片會跳過下載（基於 MD5 哈希檢測）
- 已處理的影片可直接使用，無需重新識別

### 影片儲存空間估算
- 單個手語影片：5-50 MB（依長度而定）
- 建議預留空間：2-5 GB（約 100-200 個影片）

### 手動清理影片
如需清理舊影片，可透過以下方式：

#### 方法 1：刪除特定影片
```bash
# SSH 進入伺服器
cd /opt/render/project/data/downloaded_videos  # Python 部署
# 或
cd /app/data/downloaded_videos                  # Docker 部署

# 刪除特定影片
rm <video_hash>.mp4
```

#### 方法 2：清空所有影片
```bash
# 刪除所有影片
rm -rf /opt/render/project/data/downloaded_videos/*

# 清空哈希記錄（重要！）
rm /opt/render/project/data/downloaded_videos.json
rm /opt/render/project/data/processed_count.json

# 重啟服務
```

#### 方法 3：設定自動清理（cron job）
```bash
# 每週日凌晨 2 點清理 30 天前的影片
0 2 * * 0 find /opt/render/project/data/downloaded_videos -name "*.mp4" -mtime +30 -delete
```

## 疑難排解

### 部署相關

#### 問題：Render 部署失敗，提示記憶體不足
**原因**：深度學習依賴包（TensorFlow、PyTorch）體積較大

**解決方案**：
1. 使用 Docker 部署（更穩定）
2. 升級到 Standard 或更高等級的 Instance Type
3. 如果使用 Python 部署，確保 Build 時間不超時（調整 Build Command）

#### 問題：模型載入失敗
**原因**：模型文件未正確上傳到 Git 倉庫

**解決方案**：
```bash
# 確認模型文件存在
ls -lh model_output/best_model_mps.keras
ls -lh model_output/label_map.json

# 如果文件過大（>100MB），使用 Git LFS
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add model_output/best_model_mps.keras
git commit -m "Add model with Git LFS"
git push
```

#### 問題：OpenCV 或 MediaPipe 無法載入
**原因**：缺少系統依賴

**解決方案**：
- 使用提供的 Dockerfile（已包含所需依賴）
- 或在 Render Build Command 中添加：
```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && pip install -r requirements.txt
```

### 識別相關

#### 問題：識別結果不準確
**可能原因**：
1. 影片品質不佳（模糊、光線不足）
2. 手語動作不完整或過快
3. 影片角度不正確

**建議**：
- 使用正面拍攝，確保手部和上半身清晰可見
- 每個手語動作持續 2-3 秒
- 充足光線，避免背光

#### 問題：識別速度過慢
**原因**：影片過長，窗口過多

**解決方案**：
1. 調整滑動步長（修改 `stride` 參數）
2. 升級 Render Instance Type（更多 CPU 核心）
3. 使用較短的影片（建議 10 秒以內）

#### 問題：OpenAI 句子重組失敗
**原因**：未設定 `OPENAI_API_KEY` 或 API 配額用盡

**解決方案**：
- 設定有效的 OpenAI API Key
- 或不使用 OpenAI（系統會自動使用 Top-1 單詞序列）

### Messenger Bot 相關

#### 問題：前端無法播放影片
**原因**：
- 影片路徑錯誤
- 檔案權限問題
- 瀏覽器不支援影片格式

**解決方案**：
1. 檢查瀏覽器控制台是否有 404 錯誤
2. 確認影片檔案存在：
```bash
ls -lh /opt/render/project/data/downloaded_videos/
```
3. 檢查檔案權限：
```bash
chmod 644 /opt/render/project/data/downloaded_videos/*.mp4
```

#### 問題：Messenger 無法回覆訊息
**原因**：
- `PAGE_ACCESS_TOKEN` 錯誤或過期
- 網路連線問題

**解決方案**：
1. 在 Facebook Developers Console 重新生成 Token
2. 確認 Render 伺服器能連接 `graph.facebook.com`：
```bash
curl https://graph.facebook.com/v18.0/me?access_token=YOUR_TOKEN
```
3. 查看 Render Logs 確認錯誤訊息

#### 問題：WebSocket 連線失敗
**原因**：防火牆或代理伺服器阻擋

**解決方案**：
- 確認 Render 允許 WebSocket 連線（預設支援）
- 檢查前端控制台是否有 Socket.IO 錯誤
- 嘗試重新整理頁面
- 檢查瀏覽器是否阻擋 WebSocket

### 效能優化

#### 提升識別速度
1. **調整窗口步長**：
```python
# 在 app.py 中修改
recognizer = SlidingWindowInference(
    stride=80  # 增加步長，減少窗口數量
)
```

2. **使用 GPU**（如果 Render 支援）：
```python
recognizer = SlidingWindowInference(
    device='gpu'  # 使用 GPU 加速
)
```

3. **預載入模型**：
- 系統啟動時自動載入，無需每次重新載入

#### 降低記憶體使用
1. 定期清理舊影片
2. 使用較小的批次大小
3. 關閉不必要的日誌輸出

## 開發指南

### 本地開發

```bash
# 1. Clone 專案
git clone <your-repo-url>
cd MVP拷貝2

# 2. 建立虛擬環境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 設定環境變數
cp .env.example .env
# 編輯 .env 填入你的 Token

# 5. 啟動服務
python app.py
```

### 測試手語識別

```python
from sliding_window_inference import SlidingWindowInference

# 初始化識別器
recognizer = SlidingWindowInference(
    model_path='model_output/best_model_mps.keras',
    label_map_path='model_output/label_map.json',
    device='cpu',
    stride=80,
    openai_api_key='your-api-key'  # 選填
)

# 處理影片
results = recognizer.process_video('test_video.mp4')

# 獲取句子
if recognizer.openai_client:
    sentence, explanation = recognizer.compose_sentence_with_openai(results)
    print(f"識別結果: {sentence}")
```

### 新增手語單詞

1. **準備訓練數據**：每個單詞至少 50 個影片樣本
2. **重新訓練模型**：使用 `training/train_bigrunet_tpu.py`
3. **更新標籤映射**：修改 `model_output/label_map.json`
4. **測試新模型**：確保準確率達標
5. **部署更新**：推送到 Git 倉庫，Render 自動重新部署

## 授權

MIT License

## 聯絡方式

如有問題或建議，請開 Issue 或 Pull Request。

---

**部署完成後即可使用** 🚀🤟