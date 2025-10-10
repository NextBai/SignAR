#!/bin/bash
# Render 啟動腳本 - 使用 eventlet worker

gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app
