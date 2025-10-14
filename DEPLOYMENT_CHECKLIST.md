# Render 部署檢查清單 ✅

## 部署前檢查

### 1. 必要文件
- [ ] `app.py` - 主應用程式
- [ ] `sliding_window_inference.py` - 手語識別核心
- [ ] `requirements.txt` - Python 依賴
- [ ] `Dockerfile` - Docker 配置
- [ ] `README.md` - 專案文檔
- [ ] `.gitignore` - Git 忽略文件
- [ ] `.env.example` - 環境變數範例

### 2. 模型文件（重要！）
- [ ] `model_output/best_model_mps.keras` 存在
- [ ] `model_output/label_map.json` 存在
- [ ] 模型文件大小檢查（如果 >100MB，考慮使用 Git LFS）

### 3. 特徵提取模組
- [ ] `feature_extraction/rgb_feature_extraction.py`
- [ ] `feature_extraction/skeleton_feature_extraction.py`
- [ ] `feature_extraction/__init__.py` (可選)

### 4. 前端文件
- [ ] `templates/index.html`
- [ ] `static/` 目錄（如果有）

### 5. 配置文件
- [ ] 檢查 `requirements.txt` 包含所有依賴
- [ ] 檢查 `Dockerfile` 包含系統依賴

## Render 設定檢查

### 1. Web Service 配置
- [ ] Environment: Docker 或 Python 3
- [ ] Branch: main (或你的主分支)
- [ ] Instance Type: Standard 或以上（推薦）

### 2. 環境變數設定
必需：
- [ ] `MESSENGER_VERIFY_TOKEN`
- [ ] `PAGE_ACCESS_TOKEN`

選填：
- [ ] `OPENAI_API_KEY`
- [ ] `SECRET_KEY`
- [ ] `DATA_DIR`
- [ ] `PORT`

系統變數（自動配置）：
- [ ] `KERAS_BACKEND=tensorflow`
- [ ] `TF_CPP_MIN_LOG_LEVEL=2`
- [ ] `PYTHONUNBUFFERED=1`

### 3. 持久化儲存
- [ ] 建立 Disk: `video-storage`
- [ ] Mount Path: `/opt/render/project/data` 或 `/app/data`
- [ ] Size: 至少 2GB

### 4. 資源配置
- [ ] RAM: 至少 2GB（模型需要約 1GB）
- [ ] CPU: 2 核心以上

## 部署後驗證

### 1. 服務狀態
```bash
# 檢查健康狀態
curl https://your-app.onrender.com/health

# 預期回應：
# {
#   "status": "healthy",
#   "downloaded_videos_count": 0,
#   "processed_count": 0,
#   "verify_token_set": true,
#   "page_token_set": true
# }
```

### 2. 系統資訊
```bash
# 檢查調試資訊
curl https://your-app.onrender.com/debug

# 確認：
# - model_path 存在
# - feature_extraction 模組可載入
# - 環境變數正確設定
```

### 3. Webhook 驗證
```bash
# 測試 Webhook 驗證
curl "https://your-app.onrender.com/webhook?hub.mode=subscribe&hub.verify_token=YOUR_TOKEN&hub.challenge=test123"

# 預期回應：test123
```

### 4. Render Logs 檢查
在 Render Dashboard > Logs 中確認：
- [ ] "✅ 系統就緒，等待請求..."
- [ ] "🤖 手語識別器: ✅ 已啟用"
- [ ] 無錯誤或警告訊息

## Facebook Messenger 設定

### 1. Webhook 設定
- [ ] URL: `https://your-app.onrender.com/webhook`
- [ ] Verify Token: 與 `MESSENGER_VERIFY_TOKEN` 一致
- [ ] 訂閱事件：`messages`, `messaging_postbacks`

### 2. 頁面訂閱
- [ ] 選擇正確的 Facebook 粉絲專頁
- [ ] 確認訂閱成功

### 3. 測試 Bot
- [ ] 發送文字訊息 → 收到 "請傳送手語影片給我"
- [ ] 發送手語影片 → 收到識別結果

## 常見問題快速檢查

### 部署失敗
- [ ] 檢查 Build Logs 是否有錯誤
- [ ] 確認 requirements.txt 格式正確
- [ ] 確認 Python 版本兼容（3.10）

### 模型載入失敗
- [ ] 模型文件是否正確上傳到 Git
- [ ] 檔案路徑是否正確
- [ ] 是否需要使用 Git LFS

### Bot 無回應
- [ ] PAGE_ACCESS_TOKEN 是否正確
- [ ] Webhook 是否正確設定
- [ ] 檢查 Render Logs 的錯誤訊息

### 識別結果不正確
- [ ] 影片品質是否良好
- [ ] 手語動作是否完整
- [ ] 檢查模型版本是否正確

## 效能監控

部署後持續監控：
- [ ] Render Metrics: CPU、Memory、Bandwidth
- [ ] Response Time: 平均回應時間
- [ ] Error Rate: 錯誤率
- [ ] Disk Usage: 儲存空間使用情況

## 維護建議

- [ ] 每週檢查 Disk 使用量
- [ ] 定期清理舊影片（30 天以上）
- [ ] 監控 OpenAI API 使用量（如果使用）
- [ ] 定期備份模型文件
- [ ] 更新依賴包（安全性更新）

---

**完成所有檢查項目後即可部署！** 🚀
