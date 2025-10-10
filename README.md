# 影片處理生產線 🏭

基於 Flask + WebSocket 的影片處理系統，整合 Facebook Messenger Bot，提供即時動畫展示介面。

## 功能特點

- 🎨 **生產線視覺介面**：動畫化的影片處理流程
- 📹 **Messenger Bot 整合**：接收 Facebook Messenger 傳來的影片
- 🎬 **即時影片播放**：前端自動載入並播放收到的影片
- 💾 **智慧重複檢測**：MD5 哈希值避免重複處理
- 💬 **WebSocket 即時同步**：Messenger Bot 觸發前端動畫
- 📊 **統計追蹤**：即時處理數量統計

## 使用方式

### Messenger Bot
1. 發送影片到 Facebook Messenger Bot
2. 前端即時顯示生產線動畫
3. 影片自動載入到播放器
4. Bot 自動回覆 "Hello World"

## 環境變數設定

需要設定以下環境變數：

- `MESSENGER_VERIFY_TOKEN`: Messenger Webhook 驗證令牌
- `PAGE_ACCESS_TOKEN`: Facebook Page 存取令牌
- `SECRET_KEY`: Flask Secret Key（選填，系統會自動生成）
- `DATA_DIR`: 影片儲存目錄（選填，預設 `/app/data`）
- `PORT`: 服務端口（選填，預設 `7860`）

## 技術架構

- **前端**: HTML/CSS/JavaScript + Socket.IO
- **後端**: Flask + Flask-SocketIO + Eventlet
- **部署**: Render / Docker / 任何支援 Python 的平台
- **即時通訊**: WebSocket 同步前後端動畫
- **影片儲存**: 本地檔案系統（需持久化儲存）

## 部署到 Render

### 1. 建立 Web Service
1. 登入 [Render](https://render.com/)
2. 點擊 "New +" → "Web Service"
3. 連接你的 GitHub 倉庫
4. 選擇此專案

### 2. 設定部署配置
- **Name**: `video-processing-line`（自訂）
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app`
  - 或使用腳本：`bash render_start.sh`

### 3. 設定環境變數
在 Render Dashboard > Environment 中新增：
```
MESSENGER_VERIFY_TOKEN=你的驗證令牌
PAGE_ACCESS_TOKEN=你的頁面存取令牌
DATA_DIR=/opt/render/project/data
PORT=10000
```

### 4. 啟用持久化儲存（重要！）
- 在 Render Dashboard > Disks 新增 Disk
- Mount Path: `/opt/render/project/data`
- Size: 依需求（建議至少 1GB）

### 5. 部署
點擊 "Create Web Service"，Render 會自動部署。

## Messenger Webhook 設定

1. 前往 [Facebook Developers Console](https://developers.facebook.com/)
2. 選擇你的應用 → Messenger → 設定
3. 設定 Webhook URL：`https://your-app-name.onrender.com/webhook`
4. 設定 Verify Token 為你的 `MESSENGER_VERIFY_TOKEN`
5. 訂閱 `messages` 事件
6. 選擇 Facebook 粉絲專頁並訂閱

## Docker 部署（可選）

如果使用 Docker 部署：

```bash
docker build -t video-processing-line .
docker run -d \
  -p 7860:7860 \
  -e MESSENGER_VERIFY_TOKEN=你的令牌 \
  -e PAGE_ACCESS_TOKEN=你的令牌 \
  -v $(pwd)/data:/app/data \
  video-processing-line
```

## API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 前端介面 |
| `/webhook` | GET/POST | Messenger Webhook |
| `/videos/<hash>` | GET | 取得影片檔案 |
| `/api/videos` | GET | 取得影片清單（JSON）|
| `/stats` | GET | 統計資訊 |
| `/health` | GET | 健康檢查 |
| `/debug` | GET | 調試資訊 |

## 支援的影片格式

MP4, AVI, MOV, MKV, FLV, WMV, WebM

## 影片管理

### 自動保留策略
- 所有透過 Messenger Bot 收到的影片會**保留**在伺服器上
- 影片存放於 `DATA_DIR/downloaded_videos/` 目錄
- 重複影片會跳過下載（基於 MD5 哈希檢測）

### 手動清理影片
如需清理舊影片，可透過以下方式：

**方法 1：刪除特定影片**
```bash
# SSH 進入伺服器
cd /opt/render/project/data/downloaded_videos
rm <video_hash>.mp4
```

**方法 2：清空所有影片**
```bash
rm -rf /opt/render/project/data/downloaded_videos/*
# 需同時清空哈希記錄
rm /opt/render/project/data/downloaded_videos.json
```

## 疑難排解

### 問題：前端無法播放影片
- 檢查瀏覽器控制台是否有 404 錯誤
- 確認影片檔案存在於 `DATA_DIR/downloaded_videos/`
- 檢查檔案權限（應為可讀）

### 問題：Messenger 無法回覆訊息
- 確認 `PAGE_ACCESS_TOKEN` 正確且未過期
- 檢查 Render 伺服器能否連接 `graph.facebook.com`
- 查看 Render Logs 確認錯誤訊息

### 問題：WebSocket 連線失敗
- 確認 Render 允許 WebSocket 連線（預設支援）
- 檢查前端控制台是否有 Socket.IO 錯誤
- 嘗試重新整理頁面

---

**部署完成後即可使用** 🚀