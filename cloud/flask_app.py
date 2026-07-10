import time
import requests
from flask import Flask

app = Flask(__name__)

# --- Настройки ---
TOKEN = "8668270474:AAESCzNq9Ltda_bskgVIFg-lsA5gguCKz44"  # Замените на токен вашего бота
LAST_SEEN_FILE = "last_seen.txt"

@app.route('/ping')
def ping():
    """Сюда ваш домашний ПК шлет сигнал жизни раз в минуту"""
    try:
        with open(LAST_SEEN_FILE, "w") as f:
            f.write(str(time.time()))
        return "ok"
    except Exception as e:
        return f"error: {e}", 500

@app.route('/check')
def check_status():
    """Сюда будет заходить внешний планировщик cron-job.org"""
    try:
        try:
            with open(LAST_SEEN_FILE, "r") as f:
                last_seen = float(f.read().strip())
        except Exception:
            last_seen = 0.0

        now = time.time()

        # Если пинга от домашнего ПК не было больше 3 минут (180 секунд)
        if now - last_seen > 180:
            url = f"https://api.telegram.org/bot{TOKEN}/setMyShortDescription"
            payload = {
                "short_description": "🔴 Вне сети. Домашний сервер отключен (нет электричества или интернета)."
            }
            # PythonAnywhere разрешает бесплатным аккаунтам слать запросы к api.telegram.org
            requests.post(url, json=payload, timeout=10)
            return "status: detected offline"

        return "status: online"
    except Exception as e:
        return f"check error: {e}", 500