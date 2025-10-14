#!/bin/bash
# 安全檢查腳本 - 檢查是否有硬編碼的敏感資訊

echo "🔒 安全檢查：掃描硬編碼的敏感資訊..."
echo "========================================"

# 檢查硬編碼的 API 金鑰
echo "🔑 檢查 OpenAI API 金鑰..."
if grep -r "sk-[a-zA-Z0-9]\{48\}" --include="*.py" --include="*.js" --include="*.json" .; then
    echo "❌ 發現硬編碼的 OpenAI API 金鑰！"
    exit 1
else
    echo "✅ 沒有發現硬編碼的 OpenAI API 金鑰"
fi

# 檢查硬編碼的 Facebook Token
echo "📘 檢查 Facebook Access Token..."
if grep -r "EA[A-Za-z0-9]\{20\}" --include="*.py" --include="*.js" --include="*.json" .; then
    echo "❌ 發現硬編碼的 Facebook Access Token！"
    exit 1
else
    echo "✅ 沒有發現硬編碼的 Facebook Access Token"
fi

# 檢查硬編碼的通用 Token 模式
echo "🔐 檢查通用 Token 模式..."
if grep -r "[A-Za-z0-9]\{32\}-[A-Za-z0-9]\{8\}-[A-Za-z0-9]\{4\}-[A-Za-z0-9]\{4\}-[A-Za-z0-9]\{4\}-[A-Za-z0-9]\{12\}" --include="*.py" --include="*.js" --include="*.json" . | grep -v "templates/\|static/"; then
    echo "❌ 發現硬編碼的 UUID 格式 Token！"
    exit 1
else
    echo "✅ 沒有發現硬編碼的 UUID 格式 Token"
fi

# 檢查硬編碼的密碼
echo "🔒 檢查硬編碼的密碼..."
if grep -r "password\|passwd\|credential" --include="*.py" --include="*.js" --include="*.json" . | grep -i "password\|passwd\|credential" | grep -v "templates/\|static/\|README\|requirements"; then
    echo "❌ 發現可能的硬編碼密碼！"
    exit 1
else
    echo "✅ 沒有發現硬編碼的密碼"
fi

# 檢查環境變數使用
echo "🌍 檢查環境變數使用..."
echo "檢查 MESSENGER_VERIFY_TOKEN..."
if grep -r "MESSENGER_VERIFY_TOKEN" --include="*.py" .; then
    echo "✅ 使用環境變數 MESSENGER_VERIFY_TOKEN"
else
    echo "⚠️  未找到 MESSENGER_VERIFY_TOKEN 環境變數使用"
fi

echo "檢查 PAGE_ACCESS_TOKEN..."
if grep -r "PAGE_ACCESS_TOKEN" --include="*.py" .; then
    echo "✅ 使用環境變數 PAGE_ACCESS_TOKEN"
else
    echo "⚠️  未找到 PAGE_ACCESS_TOKEN 環境變數使用"
fi

echo "檢查 OPENAI_API_KEY..."
if grep -r "OPENAI_API_KEY" --include="*.py" .; then
    echo "✅ 使用環境變數 OPENAI_API_KEY"
else
    echo "⚠️  未找到 OPENAI_API_KEY 環境變數使用"
fi

# 檢查 .env 文件
echo "📄 檢查 .env 文件..."
if [ -f ".env" ]; then
    echo "⚠️  發現 .env 文件，請確保它在 .gitignore 中"
    if grep -q ".env" .gitignore 2>/dev/null; then
        echo "✅ .env 文件已在 .gitignore 中"
    else
        echo "❌ .env 文件不在 .gitignore 中！"
        exit 1
    fi
else
    echo "✅ 沒有發現 .env 文件"
fi

# 檢查敏感文件是否在 Git 中
echo "📁 檢查敏感文件是否在 Git 中..."
if git ls-files | grep -E "\.(pem|key|crt|p12|pfx|env)$" >/dev/null 2>&1; then
    echo "❌ 發現敏感文件在 Git 中！"
    git ls-files | grep -E "\.(pem|key|crt|p12|pfx|env)$"
    exit 1
else
    echo "✅ 沒有敏感文件在 Git 中"
fi

echo ""
echo "========================================"
echo "🎉 安全檢查完成！所有檢查都通過了。"
echo ""
echo "💡 安全建議："
echo "   - 確保所有敏感資訊都通過環境變數設定"
echo "   - 不要將 .env 文件提交到 Git"
echo "   - 定期輪換 API 金鑰和 Token"
echo "   - 使用最小權限原則設定 Token 權限"