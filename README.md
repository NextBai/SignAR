---
title: Video Processing Production Line
emoji: 🏆
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
license: mit
---

# 影片處理生產線 🏭

運行在 Hugging Face Spaces 的影片處理系統，具有生產線動畫介面和 Messenger Bot 整合。

## 功能特點

- 🎨 **生產線視覺介面**：動畫化的影片處理流程
- 📹 **雙重上傳方式**：網頁前端或 Messenger Bot
- 💾 **智慧重複檢測**：MD5 哈希值避免重複處理
- 🗑️ **自動清理**：處理完成後自動刪除影片
- 💬 **即時同步**：Messenger Bot 觸發前端動畫
- 📊 **統計追蹤**：即時處理數量統計

## 使用方式

### 前端網頁
1. 訪問應用首頁
2. 上傳影片到進料口
3. 觀看生產線動畫
4. 收到 "Hello World" 訊息

### Messenger Bot
1. 發送影片到 Bot
2. 前端即時顯示動畫
3. 自動回覆 "Hello World"
4. 影片自動刪除

## 環境變數設定

在 Hugging Face Space Settings > Repository secrets 中設定：

- `MESSENGER_VERIFY_TOKEN`: Messenger Webhook 驗證令牌
- `PAGE_ACCESS_TOKEN`: Facebook Page 存取令牌
- `SECRET_KEY`: Flask Secret Key（選填，系統會自動生成）

## 技術架構

- **前端**: HTML/CSS/JavaScript + Socket.IO
- **後端**: Flask + Flask-SocketIO
- **部署**: Docker on Hugging Face Spaces
- **即時通訊**: WebSocket 同步前後端動畫

## Messenger Webhook 設定

1. 前往 Facebook Developers Console
2. 設定 Webhook URL：`https://your-space-url.hf.space/webhook`
3. 設定 Verify Token 為您的 `MESSENGER_VERIFY_TOKEN`
4. 訂閱 `messages` 事件

## 重要限制 ⚠️

### Hugging Face Spaces 網路限制

由於 Hugging Face Spaces 的安全政策，**無法直接訪問某些外部域名**，包括：
- ❌ `graph.facebook.com` - Facebook Graph API
- ❌ Meta/Facebook 相關服務

### 影響

- ✅ **前端網頁上傳**：完全正常運作
- ✅ **影片處理**：下載、檢測重複、刪除都正常
- ✅ **動畫展示**：WebSocket 即時同步正常
- ✅ **Messenger Webhook 接收**：可以接收影片
- ❌ **Messenger 訊息回覆**：無法發送（DNS 解析失敗）

### 解決方案

如需完整的 Messenger Bot 功能（包含訊息回覆），建議：
1. 使用其他平台部署（Railway, Render, Vercel, AWS, GCP）
2. 或將此 Space 作為前端展示，後端 API 部署在其他平台

## 支援的影片格式

MP4, AVI, MOV, MKV, FLV, WMV, WebM

---

**部署即自動運行，無需額外配置** 🚀