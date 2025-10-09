# 部署指南 🚀

## 快速部署到 Hugging Face Spaces

### 步驟 1: 創建 Space

1. 前往 [Hugging Face Spaces](https://huggingface.co/spaces)
2. 點擊 "Create new Space"
3. 選擇：
   - Space name: `video-processing-production-line`
   - License: `mit`
   - SDK: **Docker** ⚠️ 重要！
   - Hardware: CPU basic (免費)

### 步驟 2: 上傳檔案

上傳以下檔案到 Space：
- ✅ `app.py`
- ✅ `Dockerfile`
- ✅ `requirements.txt`
- ✅ `README.md`
- ✅ `templates/index.html`
- ✅ `.dockerignore`
- ✅ `.gitignore`

### 步驟 3: 設定環境變數

在 Space Settings > Repository secrets 新增：

```
MESSENGER_VERIFY_TOKEN=你的驗證令牌
PAGE_ACCESS_TOKEN=你的頁面存取令牌
```

> 💡 SECRET_KEY 會自動生成，無需設定

### 步驟 4: 等待部署

- Docker 會自動建置（約 2-3 分鐘）
- 看到 "Running" 狀態即表示成功
- 訪問 Space URL 測試前端

### 步驟 5: 設定 Messenger Webhook

1. 前往 [Facebook Developers](https://developers.facebook.com/)
2. 選擇你的 App
3. 進入 Messenger > Settings
4. 設定 Webhook：
   ```
   Callback URL: https://你的用戶名-video-processing-production-line.hf.space/webhook
   Verify Token: 你的 MESSENGER_VERIFY_TOKEN
   ```
5. 訂閱欄位：`messages`

## 驗證部署

### 測試前端
訪問: `https://你的用戶名-space名稱.hf.space`

### 測試 API
```bash
# 健康檢查
curl https://你的用戶名-space名稱.hf.space/health

# 統計資訊
curl https://你的用戶名-space名稱.hf.space/stats
```

### 測試 Messenger
1. 在 Messenger 發送影片給你的 Bot
2. 開啟前端頁面觀看動畫
3. 應該會即時顯示傳送帶動畫
4. 收到 "Hello World" 回覆

## 疑難排解

### Docker 建置失敗
- 檢查 Dockerfile 格式
- 確認所有依賴都在 requirements.txt

### Webhook 驗證失敗
- 確認 MESSENGER_VERIFY_TOKEN 正確
- 檢查 Space 是否正在運行
- 確認 URL 正確（包含 /webhook）

### 前端動畫不同步
- 檢查 WebSocket 連接（開啟瀏覽器 Console）
- 確認 Socket.IO 正確載入
- 重新整理頁面

### 影片上傳失敗
- 確認檔案格式（MP4, AVI, MOV 等）
- 檢查檔案大小（< 500MB）
- 查看 Space Logs

## 檔案結構

```
.
├── Dockerfile              # Docker 配置
├── requirements.txt        # Python 依賴
├── app.py                 # 主應用程式
├── templates/
│   └── index.html         # 前端介面
├── README.md              # 專案說明
├── .dockerignore          # Docker 忽略檔案
└── .gitignore            # Git 忽略檔案
```

## 自動化特性

✅ 自動建立目錄結構
✅ 自動初始化資料檔案
✅ 自動處理環境變數
✅ 自動啟動 WebSocket
✅ 自動刪除處理後的影片

**無需命令行參數，完全自動化部署！** 🎉
