import os
import io
import time
import random
import secrets
import hashlib
import threading
import signal  
import sys
import json
import re
import sqlite3
from datetime import datetime

# Отключаем GUI для Matplotlib перед импортом pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import requests
import urllib3
import numpy as np
import telebot
from telebot import types
from telebot import apihelper
from PIL import Image  # Используем Pillow для SSAA-фильтрации и экспорта

# --- Инициализация глобальных сетевых параметров ---
apihelper.CONNECT_TIMEOUT = 90
apihelper.READ_TIMEOUT = 90

# Настройка ID администратора для просмотра статистики
ADMIN_ID = int(os.environ.get("ADMIN_ID", "123456789"))  # Укажите ваш настоящий Telegram ID

# --- Инициализация графического процессора (GPU / CUDA) ---
HAS_TORCH = False
DEVICE = None

try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        torch.cuda.empty_cache()
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass
        print(f"[Device] Успешная активация GPU: {torch.cuda.get_device_name(0)}")
        print("[Device] Тензорный CUDA-движок запущен.\n")
    else:
        DEVICE = torch.device('cpu')
        print("[Device] CUDA недоступна. Вычисления переведены на CPU PyTorch.\n")
except ImportError:
    print("[Device] PyTorch не обнаружен. Вычисления переведены на CPU NumPy.\n")


# --- Потокобезопасное логирование бэкенда ---
log_lock = threading.Lock()

def log(level, section, message):
    """Выводит структурированный лог в консоль."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    with log_lock:
        print(f"[{timestamp}] [{level:<7}] [{thread_name}] [{section:<8}] {message}", flush=True)


# --- Модуль аналитики и сбора статистики (SQLite) ---
class StatsManager:
    def __init__(self, db_path="bot_stats.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)

    def _init_db(self):
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    chat_id INTEGER PRIMARY KEY,
                    join_date TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    timestamp TEXT,
                    gen_type TEXT,
                    steps INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    timestamp TEXT,
                    action TEXT
                )
            """)
            conn.commit()
            conn.close()

    def register_user(self, chat_id):
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT OR IGNORE INTO users (chat_id, join_date, is_active) VALUES (?, ?, 1)",
                (chat_id, now_str)
            )
            cursor.execute("UPDATE users SET is_active = 1 WHERE chat_id = ?", (chat_id,))
            conn.commit()
            conn.close()

    def log_generation(self, chat_id, gen_type, steps):
        self.register_user(chat_id)
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO generations (chat_id, timestamp, gen_type, steps) VALUES (?, ?, ?, ?)",
                (chat_id, now_str, gen_type, steps)
            )
            conn.commit()
            conn.close()

    def log_subscription(self, chat_id, action):
        self.register_user(chat_id)
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO subscriptions (chat_id, timestamp, action) VALUES (?, ?, ?)",
                (chat_id, now_str, action)
            )
            conn.commit()
            conn.close()

    def set_user_inactive(self, chat_id):
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET is_active = 0 WHERE chat_id = ?", (chat_id,))
            conn.commit()
            conn.close()

    def get_weekly_report(self):
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            active_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM generations")
            total_gens = cursor.fetchone()[0]
            
            cursor.execute("SELECT gen_type, COUNT(*) FROM generations GROUP BY gen_type")
            gen_distribution = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT date(join_date), COUNT(*) 
                FROM users 
                WHERE join_date >= datetime('now', '-7 days')
                GROUP BY date(join_date)
                ORDER BY date(join_date) ASC
            """)
            user_growth_7d = cursor.fetchall()
            
            conn.close()
            
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_generations": total_gens,
            "gen_distribution": gen_distribution,
            "user_growth_7d": user_growth_7d
        }

stats = StatsManager()

# --- СЛОВАРЬ ЛОКАЛИЗАЦИИ (РУССКИЙ / ENGLISH) ---
# --- СЛОВАРЬ ЛОКАЛИЗАЦИИ (РУССКИЙ / ENGLISH) ---
TRANSLATIONS = {
    "ru": {
        "welcome": (
            "«Однажды погрузившись в фрактал, ты больше никогда не остановишься. "
            "Позволь математике растворить тебя в бесконечности иррациональных чисел...»\n\n"
            "👁‍⚙ **Синхронизация интерфейса завершена.** Используйте панель управления ниже.\n"
            "Нажмите ❓ *Помощь*, чтобы узнать подробнее о всех функциях."
        ),
        "help": HELP_TEXT_RU,  # Здесь автоматически подставится русский текст справки
        "btn_shallow": "🌌 Малые расстояния",
        "btn_deep": "🌀 Сверхглубокий зум",
        "btn_sub_start": "🧿 Запустить бесконечный поток",
        "btn_sub_stop": "⏳ Остановить бесконечный поток",
        "btn_batch3": "🔮 Сгенерировать пакет из 3 фракталов",
        "btn_batch5": "🔮 Сгенерировать пакет из 5 фракталов",
        "btn_custom": "✍️ Свой фрактал",
        "btn_settings": "⚙️ Настройки",
        "btn_help": "❓ Помощь",
        "settings_title": "⚙️ **Настройки фрактальной генерации**",
        "settings_desc": "Текущий режим пакетного рендеринга: **{mode}**\n└ _Этот режим определяет глубину зума при создании пакетов из 3 или 5 фракталов._",
        "settings_lang": "🌍 Язык интерфейса: **Русский**",
        "btn_toggle_zoom": "🔄 Переключить масштаб пакетов",
        "btn_toggle_lang": "🌍 Change Language / Сменить язык",
        "lang_changed": "Language changed to English!",
        "busy": "⚠️ Вычисления уже запущены...",
        "cooldown": "⏳ Подождите {val:.1f} сек...",
        "status_shallow": "🌌 Малые расстояния на {steps} шагов...",
        "status_deep": "🌀 Сверхглубокий масштаб на {steps} шагов...",
        "attempt": "🧬 *Попытка {attempt}/{max_attempts}*",
        "init_grid": "🧬 *Попытка {attempt}/{max_attempts}*\n└ Инициализация матрицы...",
        "vectors": "🌀 Расчёт векторов комплексного поля...",
        "zoom_position": "🪐 Позиционирование зума (Шаг {step}/{steps})...",
        "check_aesthetic": "⚡ Проверка эстетического потенциала...",
        "rendering_high": "🧬 Рендеринг фрактала высокой точности ({res}x{res})...",
        "rejected_attempt": "⚠️ *Попытка {attempt}/{max_attempts}* отклонена.\n└ _Точка поля не рекомендована к визуализации. Ищем новую область..._",
        "timeout_retry": "⚠️ *Попытка {attempt}/{max_attempts}* прервана по таймауту. Пересчет сингулярности...",
        "unstable_chaos": "👁‍⚙ Математический хаос оказался слишком неустойчив. Повторите попытку.",
        "fractal_ready": "🔮 **Погружение совершено.**\n\nХаос упорядочен формулой:\n`{formula}`\n\nКоординаты:\n`xmin = {xmin:.10f}`\n`xmax = {xmax:.10f}`\n`ymin = {ymin:.10f}`\n`ymax = {ymax:.10f}`\n\n{phrase}",
        "png_caption": "🖼️ **Оригинальная проекция (PNG, без сжатия)**\n└ _Скачайте файл, чтобы рассмотреть микродетали без артефактов._",
        "timeout_error": (
            "⚠️ **Вычисления прерваны по таймауту (2 минуты).**\n\n"
            "Генерируемое уравнение оказалось слишком ресурсоемким. "
            "Сессия была перезапущена во избежание перегрузки сервера."
        ),
        "gen_error": "❌ Произошел сбой при генерации фрактала: {error}",
        "sub_started": "👁‍⚙ **Поток запущен.**\n\nКаждые два часа математическое ядро будет автоматически проецировать новую случайную структуру высокой точности.",
        "sub_stopped": "⏳ **Поток приостановлен.**\n\nБесконечность отпускает вас... до следующего ручного погружения.",
        "batch_init": "🧬 Инициация пакетного рендеринга ({num} проекций).\nРежим масштабирования: **{mode}**.\nРасчёты проводятся последовательно.",
        "batch_step": "🪐 *Фрактал {index} из {num}*\n└ Попытка {attempt}/{max_attempts}: {text}",
        "batch_ready": "✨ **Фрактальный слой #{index}**\n\nУравнение эволюции:\n`{formula}`\n\nКоординаты:\n`xmin = {xmin:.8f}`\n`xmax = {xmax:.8f}`\n`ymin = {ymin:.8f}`\n`ymax = {ymax:.8f}`",
        "batch_failed": "❌ Не удалось пробиться сквозь хаос. Проекции не построены.",
        "batch_success": "🔮 **Пакетный перенос завершен.** Все проекции визуализированы.",
        "broadcast_caption": (
            "👁‍🗨 **Плановая материализация хаоса**\n\n"
            "Высший математический порядок пробил бесконечность.\n"
            "Проекция уравнения:\n`{formula}`\n\n"
            "Координаты:\n"
            "`xmin = {xmin:.10f}`\n"
            "`xmax = {xmax:.10f}`\n"
            "`ymin = {ymin:.10f}`\n"
            "`ymax = {ymax:.10f}`\n\n"
            "⏳ _Вы можете отключить поток в меню кнопкой в любой момент._"
        ),
        "custom_info": (
            "✍️ **Генерация собственного фрактала по формуле**\n\n"
            "Задайте формулу в алгебраическом формате и (опционально) координаты.\n\n"
            "**Пример ввода:**\n"
            "```\n"
            "Формула: (Z^2) + C\n"
            "Координаты: -2.0, 2.0, -2.0, 2.0\n"
            "```\n"
            "или просто формула (координаты определятся автозумом):\n"
            "```\n"
            "Формула: cos(Z) * C\n"
            "```\n"
            "**Переменные:** `Z`, `C` (комплексные числа)\n"
            "**Функции:** `sin`, `cos`, `exp`, `ln`, `abs`, `conj`, `inv`, `sigmoid`\n"
            "**Операторы:** `+`, `-`, `*`, `/`, `^` (степень)\n\n"
            "Пришлите ваши параметры ответным сообщением (reply) на это сообщение."
        ),
        "custom_error": "❌ **Ошибка разбора формулы:**\n`{error}`\n\nПопробуйте еще раз. Пример корректной записи: `(Z^2) + C`",
        "custom_success": (
            "🎨 **Ваш кастомный фрактал готов!**\n\n"
            "Формула:\n`{formula}`\n\n"
            "Координаты:\n"
            "`xmin = {xmin:.10f}`\n"
            "`xmax = {xmax:.10f}`\n"
            "`ymin = {ymin:.10f}`\n"
            "`ymax = {ymax:.10f}`"
        )
    },
    "en": {
        "welcome": (
            "“Once you dive into a fractal, you will never stop. "
            "Let mathematics dissolve you into the infinity of irrational numbers...”\n\n"
            "👁‍⚙ **Interface synchronization complete.** Use the control panel below.\n"
            "Press ❓ *Help* to learn more about all the features."
        ),
        "help": HELP_TEXT_EN,  # Здесь автоматически подставится английский текст справки
        "btn_shallow": "🌌 Shallow Zoom",
        "btn_deep": "🌀 Deep Zoom",
        "btn_sub_start": "🧿 Start Infinite Stream",
        "btn_sub_stop": "⏳ Stop Infinite Stream",
        "btn_batch3": "🔮 Generate Batch of 3",
        "btn_batch5": "🔮 Generate Batch of 5",
        "btn_custom": "✍️ Custom Fractal",
        "btn_settings": "⚙️ Settings",
        "btn_help": "❓ Help",
        "settings_title": "⚙️ **Fractal Generation Settings**",
        "settings_desc": "Current batch rendering mode: **{mode}**\n└ _This mode determines the zoom depth when creating batches of 3 or 5 fractals._",
        "settings_lang": "🌍 Interface Language: **English**",
        "btn_toggle_zoom": "🔄 Toggle Batch Scale",
        "btn_toggle_lang": "🌍 Change Language / Сменить язык",
        "lang_changed": "Язык изменен на Русский!",
        "busy": "⚠️ Computation is already running...",
        "cooldown": "⏳ Please wait {val:.1f} sec...",
        "status_shallow": "🌌 Shallow zoom for {steps} steps...",
        "status_deep": "🌀 Deep zoom for {steps} steps...",
        "attempt": "🧬 *Attempt {attempt}/{max_attempts}*",
        "init_grid": "🧬 *Attempt {attempt}/{max_attempts}*\n└ Initializing grid...",
        "vectors": "🌀 Computing complex field vectors...",
        "zoom_position": "🪐 Positioning zoom (Step {step}/{steps})...",
        "check_aesthetic": "⚡ Evaluation of aesthetic potential...",
        "rendering_high": "🧬 Rendering high-precision fractal ({res}x{res})...",
        "rejected_attempt": "⚠️ *Attempt {attempt}/{max_attempts}* rejected.\n└ _The region is not recommended for visualization. Searching for a new area..._",
        "timeout_retry": "⚠️ *Attempt {attempt}/{max_attempts}* timed out. Recalculating singularity...",
        "unstable_chaos": "👁‍⚙ Mathematical chaos proved too unstable. Please try again.",
        "fractal_ready": "🔮 **Dive completed.**\n\nChaos is ordered by formula:\n`{formula}`\n\nCoordinates:\n`xmin = {xmin:.10f}`\n`xmax = {xmax:.10f}`\n`ymin = {ymin:.10f}`\n`ymax = {ymax:.10f}`\n\n{phrase}",
        "png_caption": "🖼️ **Original projection (PNG, uncompressed)**\n└ _Download this file to see microscopic details without compression artifacts._",
        "timeout_error": (
            "⚠️ **Computation timed out (2 minutes).**\n\n"
            "The generated equation was too resource-intensive. "
            "The session was restarted to prevent server overload."
        ),
        "gen_error": "❌ Generation failed: {error}",
        "sub_started": "👁‍⚙ **Stream started.**\n\nEvery two hours, the mathematical core will automatically project a new high-precision structure to you.",
        "sub_stopped": "⏳ **Stream paused.**\n\nInfinity lets you go... until your next manual dive.",
        "batch_init": "🧬 Initiating batch rendering ({num} projections).\nZoom mode: **{mode}**.\nCalculations are performed sequentially.",
        "batch_step": "🪐 *Fractal {index} of {num}*\n└ Attempt {attempt}/{max_attempts}: {text}",
        "batch_ready": "✨ **Fractal Layer #{index}**\n\nEvolution equation:\n`{formula}`\n\nCoordinates:\n`xmin = {xmin:.8f}`\n`xmax = {xmax:.8f}`\n`ymin = {ymin:.8f}`\n`ymax = {ymax:.8f}`",
        "batch_failed": "❌ Failed to break through chaos. Projections were not rendered.",
        "batch_success": "🔮 **Batch transfer complete.** All projections rendered successfully.",
        "broadcast_caption": (
            "👁‍🗨 **Scheduled Materialization of Chaos**\n\n"
            "Higher mathematical order has pierced infinity.\n"
            "Equation projection:\n`{formula}`\n\n"
            "Coordinates:\n"
            "`xmin = {xmin:.10f}`\n"
            "`xmax = {xmax:.10f}`\n"
            "`ymin = {ymin:.10f}`\n"
            "`ymax = {ymax:.10f}`\n\n"
            "⏳ _You can stop the stream at any time using the button in the settings menu._"
        ),
        "custom_info": (
            "✍️ **Generate Your Own Fractal by Formula**\n\n"
            "Define the formula in algebraic format and (optionally) coordinates.\n\n"
            "**Input Example:**\n"
            "```\n"
            "Formula: (Z^2) + C\n"
            "Coordinates: -2.0, 2.0, -2.0, 2.0\n"
            "```\n"
            "or just the formula (coordinates will be auto-focused):\n"
            "```\n"
            "Formula: cos(Z) * C\n"
            "```\n"
            "**Variables:** `Z`, `C` (complex numbers)\n"
            "**Functions:** `sin`, `cos`, `exp`, `ln`, `abs`, `conj`, `inv`, `sigmoid`\n"
            "**Operators:** `+`, `-`, `*`, `/`, `^` (power)\n\n"
            "Send your parameters in a reply to this message."
        ),
        "custom_error": "❌ **Formula parsing error:**\n`{error}`\n\nTry again. Example of valid entry: `(Z^2) + C`",
        "custom_success": (
            "🎨 **Your custom fractal is ready!**\n\n"
            "Formula:\n`{formula}`\n\n"
            "Coordinates:\n"
            "`xmin = {xmin:.10f}`\n"
            "`xmax = {xmax:.10f}`\n"
            "`ymin = {ymin:.10f}`\n"
            "`ymax = {ymax:.10f}`"
        )
    }
}

# Английские фразы-комментарии (в дополнение к bot_phrases)
DEFAULT_PHRASES_EN = [
    "The aesthetics of fractal composition in its pure mathematical form.",
    "The balance of symmetry and asymmetry born of a formula.",
    "Geometry as a way to order visual chaos.",
    "Exploring the plastics and rhythm of complex space."
]

def get_user_lang(message_or_chat_id):
    """Определяет язык пользователя: сохраненный в JSON или по коду языка Telegram."""
    if isinstance(message_or_chat_id, (int, str)):
        chat_id = message_or_chat_id
        lang_code = "en"
    else:
        chat_id = message_or_chat_id.chat.id
        lang_code = message_or_chat_id.from_user.language_code if message_or_chat_id.from_user else "en"
        
    saved_lang = get_user_setting(chat_id, "lang", None)
    if saved_lang:
        return saved_lang
        
    if lang_code and lang_code.lower().startswith("ru"):
        return "ru"
    return "en"

# --- Сетевой слой с защитой от SSLEOFError и разрывов соединений ---
def safe_api_call(func, *args, **kwargs):
    """Выполняет вызов Telegram API с экспоненциальной задержкой при сбоях сети."""
    retries = 3
    backoff = 2.0
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(err in err_str for err in ["ssl", "connection", "timeout", "eof", "broken pipe", "max retries"])
            if is_transient and attempt < retries - 1:
                sleep_time = backoff ** (attempt + 1)
                log("WARN", "NETWORK", f"Временный сбой API ({e}). Повтор через {sleep_time:.1f}с...")
                time.sleep(sleep_time)
                continue
            raise e

def safe_send_photo(chat_id, photo, **kwargs):
    return safe_api_call(bot.send_photo, chat_id, photo, **kwargs)

def safe_send_document(chat_id, document, **kwargs):
    return safe_api_call(bot.send_document, chat_id, document, **kwargs)

def safe_edit_message_text(text, chat_id, message_id, **kwargs):
    return safe_api_call(bot.edit_message_text, text, chat_id, message_id, **kwargs)


# --- Класс плавного обновления статуса в Telegram (Защита от HTTP 429) ---
class ProgressUpdater:
    def __init__(self, bot, chat_id, message_id):
        self.bot = bot
        self.chat_id = chat_id
        self.message_id = message_id
        self.last_text = ""
        self.last_update_time = 0.0
        self.lock = threading.Lock()

    def update(self, text, force=False):
        """Редактирует сообщение в Telegram с троттлингом 1.2 сек."""
        with self.lock:
            now = time.time()
            if text == self.last_text:
                return
            
            if force or (now - self.last_update_time >= 1.2):
                try:
                    safe_edit_message_text(text, self.chat_id, self.message_id, parse_mode='Markdown')
                    self.last_text = text
                    self.last_update_time = now
                except Exception:
                    pass


# --- Настройки токена ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    BOT_TOKEN = "ВАШ_ТОКЕН_СЮДА" 

if BOT_TOKEN == "ВАШ_ТОКЕН_СЮДА" or not BOT_TOKEN:
    raise ValueError("Замените заглушку 'ВАШ_ТОКЕН_СЮДА' на реальный токен от @BotFather!")

bot = telebot.TeleBot(BOT_TOKEN)

# --- Хранилище подписок с защитой от Race Conditions ---
SUBSCRIBERS_FILE = "subscribers.txt"
subscribers_lock = threading.Lock()

def load_subscribers():
    if not os.path.exists(SUBSCRIBERS_FILE):
        return set()
    with subscribers_lock:
        try:
            with open(SUBSCRIBERS_FILE, "r") as f:
                return set(int(line.strip()) for line in f if line.strip().isdigit())
        except Exception as e:
            log("ERROR", "STORAGE", f"Ошибка чтения подписчиков: {e}")
            return set()

def save_subscriber(chat_id):
    with subscribers_lock:
        subs = set()
        if os.path.exists(SUBSCRIBERS_FILE):
            try:
                with open(SUBSCRIBERS_FILE, "r") as f:
                    subs = set(int(line.strip()) for line in f if line.strip().isdigit())
            except Exception:
                pass
        if chat_id not in subs:
            try:
                with open(SUBSCRIBERS_FILE, "a") as f:
                    f.write(f"{chat_id}\n")
                log("SUCCESS", "STORAGE", f"Пользователь {chat_id} добавлен в базу подписок.")
            except Exception as e:
                log("ERROR", "STORAGE", f"Не удалось сохранить подписчика {chat_id}: {e}")

def remove_subscriber(chat_id):
    with subscribers_lock:
        if not os.path.exists(SUBSCRIBERS_FILE):
            return
        try:
            subs = set()
            with open(SUBSCRIBERS_FILE, "r") as f:
                subs = set(int(line.strip()) for line in f if line.strip().isdigit())
            if chat_id in subs:
                subs.remove(chat_id)
                with open(SUBSCRIBERS_FILE, "w") as f:
                    for sub in subs:
                        f.write(f"{sub}\n")
                log("SUCCESS", "STORAGE", f"Пользователь {chat_id} удален из базы подписок.")
        except Exception as e:
            log("ERROR", "STORAGE", f"Не удалось удалить подписчика {chat_id}: {e}")


# --- Менеджер настроек пользователей ---
SETTINGS_FILE = "user_settings.json"
settings_lock = threading.Lock()

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return {}
    with settings_lock:
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

def save_user_setting(chat_id, key, value):
    with settings_lock:
        settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except Exception:
                pass
        str_id = str(chat_id)
        if str_id not in settings:
            settings[str_id] = {}
        settings[str_id][key] = value
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log("ERROR", "STORAGE", f"Не удалось сохранить настройки для {chat_id}: {e}")

def get_user_setting(chat_id, key, default):
    settings = load_settings()
    return settings.get(str(chat_id), {}).get(key, default)


# --- Менеджер состояний пользователей ---
class UserManager:
    def __init__(self):
        self.active_jobs = set()  
        self.last_request = {}    
        self.lock = threading.Lock()
        
    def try_start_job(self, chat_id, cooldown_sec=4.0):
        with self.lock:
            now = time.time()
            last_time = self.last_request.get(chat_id, 0.0)
            if now - last_time < cooldown_sec:
                log("WARN", "USER", f"Юзер {chat_id} спамит кнопками. Отказ (кулдаун: {cooldown_sec - (now - last_time):.1f}с)")
                return "cooldown", cooldown_sec - (now - last_time)
            
            if chat_id in self.active_jobs:
                log("WARN", "USER", f"Юзер {chat_id} попытался запустить вычисления параллельно. Отказ.")
                return "busy", None
            
            self.active_jobs.add(chat_id)
            self.last_request[chat_id] = now
            return "ok", None
            
    def end_job(self, chat_id):
        with self.lock:
            if chat_id in self.active_jobs:
                self.active_jobs.remove(chat_id)

user_manager = UserManager()


# --- Система случайных фраз ---
PHRASES_FILE = "phrases.txt"

DEFAULT_PHRASES = [
    "Эстетика фрактальной композиции в ее чистом математическом проявлении.",
    "Баланс симметрии и асимметрии, рожденный формулой.",
    "Геометрия как способ упорядочить визуальный хаос.",
    "Исследование пластики и ритма комплексного пространства."
]

def load_phrases():
    if not os.path.exists(PHRASES_FILE):
        log("WARN", "SYSTEM", f"Файл {PHRASES_FILE} не найден. Используются встроенные фразы.")
        return DEFAULT_PHRASES
    try:
        with open(PHRASES_FILE, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]
        if phrases:
            log("INFO", "SYSTEM", f"Успешно загружено {len(phrases)} фраз для генератора.")
            return phrases
        return DEFAULT_PHRASES
    except Exception as e:
        log("ERROR", "SYSTEM", f"Ошибка при чтении {PHRASES_FILE}: {e}")
        return DEFAULT_PHRASES

bot_phrases = load_phrases()

def get_random_phrase():
    return random.choice(bot_phrases)


# --- Константы RPN-вычислителя ---
VAR_Z, VAR_C, CONST = 0, 1, 2
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW = 3, 4, 5, 6, 7
OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM = 8, 9, 10, 11, 12, 13, 14, 15

EPS_REG = 1e-20

# --- Интегральный фильтр для локализации фрактальной границы ---
def fast_uniform_filter(arr, size=15):
    sz = size // 2
    padded = np.pad(arr, sz, mode='edge')
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumsum_padded = np.pad(cumsum, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    h, w = arr.shape
    total = (cumsum_padded[size:h + size, size:w + size]
             - cumsum_padded[0:h, size:w + size]
             - cumsum_padded[size:h + size, 0:w]
             + cumsum_padded[0:h, 0:w])
    return total / (size * size)

def get_classic_colormap():
    colors = [
        (0.0, '#000105'), (0.12, '#01061c'), (0.32, '#041d5e'),
        (0.55, '#2269eb'), (0.72, '#82b5ff'), (0.85, '#ffffff'),
        (0.92, '#ffaa00'), (0.97, '#ff3700'), (1.0, '#000000')
    ]
    return LinearSegmentedColormap.from_list("DynamicMap", colors, N=2048)

CLASSIC_CMAP = get_classic_colormap()


# --- Декодирование процедурной грамматики ---
class EntropyDecoder:
    def __init__(self, seed_int: int):
        self.state = seed_int.to_bytes(64, 'big')
        self.buffer = []
        self.pointer = 0

    def _refresh_entropy(self):
        self.state = hashlib.sha512(self.state).digest()
        self.buffer.extend(list(self.state))
        self.pointer = 0

    def get_next_byte(self) -> int:
        if not self.buffer or self.pointer >= len(self.buffer):
            self._refresh_entropy()
        val = self.buffer[self.pointer]
        self.pointer += 1
        return val

    def get_float(self) -> float:
        b0 = self.get_next_byte()
        b1 = self.get_next_byte()
        b2 = self.get_next_byte()
        b3 = self.get_next_byte()
        val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        return val / 4294967296.0

def generate_ast(decoder, depth, max_depth):
    # Ограничиваем глубину дерева до 4, чтобы формулы оставались элегантными и сходящимися
    p_terminal = min(1.0, (depth - 1) / (max_depth - 1)) if depth > 1 else 0.0
    if decoder.get_float() < p_terminal:
        r = decoder.get_float()
        if r < 0.45: return {"type": "terminal", "opcode": VAR_Z, "val": None}
        elif r < 0.85: return {"type": "terminal", "opcode": VAR_C, "val": None}
        else:
            real = -1.5 + 3.0 * decoder.get_float()
            imag = -1.5 + 3.0 * decoder.get_float()
            return {"type": "terminal", "opcode": CONST, "val": complex(real, imag)}
    else:
        r_op = decoder.get_float()
        if r_op < 0.35: # Unary
            op_list = [OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            child = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "unary", "opcode": op, "child": child}
        else: # Binary
            # Дублируем умножение и сложение, чтобы ослабить деление и степени (снижает хаотичный шум)
            op_list = [OP_ADD, OP_SUB, OP_MUL, OP_MUL, OP_DIV, OP_POW]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            left = generate_ast(decoder, depth + 1, max_depth)
            right = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "binary", "opcode": op, "left": left, "right": right}

def ast_to_rpn(node, rpn):
    if node["type"] == "terminal":
        rpn.append((0, node["opcode"], node["val"]))
    elif node["type"] == "unary":
        ast_to_rpn(node["child"], rpn)
        rpn.append((1, node["opcode"], None))
    elif node["type"] == "binary":
        ast_to_rpn(node["left"], rpn)
        ast_to_rpn(node["right"], rpn)
        rpn.append((2, node["opcode"], None))

def validate_rpn(rpn):
    has_z = any(t == 0 and op == VAR_Z for t, op, _ in rpn)
    has_c = any(t == 0 and op == VAR_C for t, op, _ in rpn)
    num_ops = sum(1 for t, _, _ in rpn if t in (1, 2))
    return has_z and has_c and (num_ops >= 1)

def rpn_to_str(rpn):
    stack = []
    op_syms = {OP_ADD: "+", OP_SUB: "-", OP_MUL: "*", OP_DIV: "/", OP_POW: "^"}
    unary_names = {
        OP_SIN: "sin", OP_COS: "cos", OP_EXP: "exp", OP_LN: "ln",
        OP_ABS: "abs_rect", OP_CONJ: "conj", OP_INV: "inv", OP_SIGM: "sigmoid"
    }
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z: stack.append("Z")
            elif op == VAR_C: stack.append("C")
            elif op == CONST: stack.append(f"({val.real:.2f}+{val.imag:.2f}j)")
        elif t_type == 1:
            arg = stack.pop()
            stack.append(f"{unary_names[op]}({arg})")
        elif t_type == 2:
            arg2 = stack.pop()
            arg1 = stack.pop()
            stack.append(f"({arg1}{op_syms[op]}{arg2})")
    return stack[0] if stack else "Z"


# --- ПАРСЕР АЛГЕБРАИЧЕСКИХ ВЫРАЖЕНИЙ В RPN (Алгоритм Сортировочной Станции) ---
def parse_infix_to_rpn(expr_str):
    """Преобразует строку формулы пользователя в RPN-представление."""
    expr_str = expr_str.replace(" ", "").replace("abs_rect", "abs")
    
    token_specification = [
        ('COMPLEX', r'\d+(?:\.\d*)?[jJ]|\d+(?:\.\d*)?[\+\-]\d+(?:\.\d*)?[jJ]'), 
        ('NUMBER',  r'\d+(?:\.\d*)?'),                                       
        ('IDENT',   r'[a-zA-Z_][a-zA-Z0-9_]*'),                              
        ('OP',      r'[\+\-\*\/\^\(\),]')                                    
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    tokens = []
    for mo in re.finditer(tok_regex, expr_str):
        tokens.append((mo.lastgroup, mo.group()))
    
    output = []
    op_stack = []
    
    prec = {'+': 2, '-': 2, '*': 3, '/': 3, '^': 4}
    assoc = {'+': 'L', '-': 'L', '*': 'L', '/': 'L', '^': 'R'}
    funcs = {'sin', 'cos', 'exp', 'ln', 'abs', 'conj', 'inv', 'sigmoid'}
    
    prev_token_type = 'START'
    
    for kind, val in tokens:
        if kind == 'NUMBER' or kind == 'COMPLEX':
            c_val = complex(val.replace('j', 'j').replace('J', 'j')) if 'j' in val.lower() else complex(float(val))
            output.append((0, CONST, c_val))
            prev_token_type = 'NUMBER'
        elif kind == 'IDENT':
            val_lower = val.lower()
            if val_lower == 'z':
                output.append((0, VAR_Z, None))
                prev_token_type = 'IDENT'
            elif val_lower == 'c':
                output.append((0, VAR_C, None))
                prev_token_type = 'IDENT'
            elif val_lower in funcs:
                op_stack.append(('FUNC', val_lower))
                prev_token_type = 'IDENT'
            else:
                raise ValueError(f"Неизвестная функция или переменная: {val}")
        elif val == '(':
            op_stack.append(('PAREN', '('))
            prev_token_type = 'LPAREN'
        elif val == ')':
            while op_stack and op_stack[-1][1] != '(':
                output.append(op_stack.pop())
            if not op_stack:
                raise ValueError("Несогласованные скобки")
            op_stack.pop() 
            if op_stack and op_stack[-1][0] == 'FUNC':
                output.append(op_stack.pop())
            prev_token_type = 'RPAREN'
        elif val == ',':
            while op_stack and op_stack[-1][1] != '(':
                output.append(op_stack.pop())
            prev_token_type = 'COMMA'
        elif kind == 'OP':
            is_unary = False
            if val in ('+', '-'):
                if prev_token_type in ('START', 'LPAREN', 'COMMA', 'OP'):
                    is_unary = True
            
            if is_unary:
                if val == '-':
                    output.append((0, CONST, 0j))
                    while (op_stack and op_stack[-1][0] == 'OP' and 
                           (assoc['-'] == 'L' and prec['-'] <= prec.get(op_stack[-1][1], 0) or
                            assoc['-'] == 'R' and prec['-'] < prec.get(op_stack[-1][1], 0))):
                        output.append(op_stack.pop())
                    op_stack.append(('OP', '-'))
            else:
                while (op_stack and op_stack[-1][0] == 'OP' and 
                       (assoc[val] == 'L' and prec[val] <= prec.get(op_stack[-1][1], 0) or
                        assoc[val] == 'R' and prec[val] < prec.get(op_stack[-1][1], 0))):
                    output.append(op_stack.pop())
                op_stack.append(('OP', val))
            prev_token_type = 'OP'
            
    while op_stack:
        top_type, top_val = op_stack.pop()
        if top_val in ('(', ')'):
            raise ValueError("Несогласованные скобки")
        output.append((top_type, top_val))
        
    rpn_tokens = []
    for item_type, item_val in output:
        if item_type == 0:
            rpn_tokens.append(item_val)
        elif item_type == 'OP':
            op_code = {'+': OP_ADD, '-': OP_SUB, '*': OP_MUL, '/': OP_DIV, '^': OP_POW}[item_val]
            rpn_tokens.append((2, op_code, None))
        elif item_type == 'FUNC':
            op_code = {
                'sin': OP_SIN, 'cos': OP_COS, 'exp': OP_EXP, 'ln': OP_LN,
                'abs': OP_ABS, 'conj': OP_CONJ, 'inv': OP_INV, 'sigmoid': OP_SIGM
            }[item_val]
            rpn_tokens.append((1, op_code, None))
    return rpn_tokens


# --- Вычислительные интерпретаторы (с поддержкой динамической точности) ---
def evaluate_rpn_pytorch(rpn, Z, C, device, use_double=False):
    stack = []
    torch_complex = torch.complex128 if use_double else torch.complex64
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z: stack.append(Z)
            elif op == VAR_C: stack.append(C)
            elif op == CONST: stack.append(torch.tensor(val, dtype=torch_complex, device=device))
        elif t_type == 1:
            A = stack.pop()
            if op == OP_SIN:
                stack.append(torch.sin(torch.complex(A.real, torch.clamp(A.imag, -15.0, 15.0))))
            elif op == OP_COS:
                stack.append(torch.cos(torch.complex(A.real, torch.clamp(A.imag, -15.0, 15.0))))
            elif op == OP_EXP:
                stack.append(torch.exp(torch.complex(torch.clamp(A.real, -15.0, 15.0), A.imag)))
            elif op == OP_LN:
                mag_sq = A.real**2 + A.imag**2 + EPS_REG
                stack.append(torch.complex(0.5 * torch.log(mag_sq), torch.atan2(A.imag, A.real)))
            elif op == OP_ABS:
                stack.append(torch.complex(torch.abs(A.real), torch.abs(A.imag)))
            elif op == OP_CONJ:
                stack.append(torch.conj(A))
            elif op == OP_INV:
                denom = A.real**2 + A.imag**2 + EPS_REG
                stack.append(torch.complex(A.real / denom, -A.imag / denom))
            elif op == OP_SIGM:
                real_clamped = torch.clamp(A.real, -15.0, 15.0)
                denom = 1.0 + torch.exp(-real_clamped) * torch.complex(torch.cos(-A.imag), torch.sin(-A.imag))
                d_mag_sq = denom.real**2 + denom.imag**2 + EPS_REG
                stack.append(torch.complex(denom.real / d_mag_sq, -denom.imag / d_mag_sq))
        elif t_type == 2:
            B = stack.pop()
            A = stack.pop()
            if op == OP_ADD: stack.append(A + B)
            elif op == OP_SUB: stack.append(A - B)
            elif op == OP_MUL: stack.append(A * B)
            elif op == OP_DIV:
                denom = B.real**2 + B.imag**2 + EPS_REG
                stack.append(torch.complex((A.real * B.real + A.imag * B.imag) / denom, (A.imag * B.real - A.real * B.imag) / denom))
            elif op == OP_POW:
                mag_sq = A.real**2 + A.imag**2 + EPS_REG
                ln_A = torch.complex(0.5 * torch.log(mag_sq), torch.atan2(A.imag, A.real))
                prod = B * ln_A
                stack.append(torch.exp(torch.complex(torch.clamp(prod.real, -15.0, 15.0), prod.imag)))
                
    Z_next = stack[0]
    anomalies = torch.isnan(Z_next) | torch.isinf(Z_next)
    if torch.any(anomalies):
        Z_next = torch.where(anomalies, torch.tensor(1e5 + 0j, dtype=torch_complex, device=device), Z_next)
    return Z_next

def evaluate_rpn_numpy(rpn, Z, C, use_double=False):
    stack = []
    complex_dtype = np.complex128 if use_double else np.complex64
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z: stack.append(Z)
            elif op == VAR_C: stack.append(C)
            elif op == CONST: stack.append(complex_dtype(val))
        elif t_type == 1:
            A = stack.pop()
            with np.errstate(invalid='ignore', over='ignore'):
                if op == OP_SIN:
                    stack.append(np.sin(np.real(A) + 1j * np.clip(np.imag(A), -15.0, 15.0)))
                elif op == OP_COS:
                    stack.append(np.cos(np.real(A) + 1j * np.clip(np.imag(A), -15.0, 15.0)))
                elif op == OP_EXP:
                    stack.append(np.exp(np.clip(np.real(A), -15.0, 15.0) + 1j * np.imag(A)))
                elif op == OP_LN:
                    mag_sq = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    stack.append(0.5 * np.log(mag_sq) + 1j * np.arctan2(np.imag(A), np.real(A)))
                elif op == OP_ABS:
                    stack.append(np.abs(np.real(A)) + 1j * np.abs(np.imag(A)))
                elif op == OP_CONJ:
                    stack.append(np.conj(A))
                elif op == OP_INV:
                    denom = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    stack.append(np.real(A)/denom - 1j*np.imag(A)/denom)
                elif op == OP_SIGM:
                    real_clamped = np.clip(np.real(A), -15.0, 15.0)
                    denom = 1.0 + np.exp(-real_clamped) * (np.cos(-np.imag(A)) + 1j * np.sin(-np.imag(A)))
                    d_mag_sq = np.real(denom)**2 + np.imag(denom)**2 + EPS_REG
                    stack.append(np.real(denom)/d_mag_sq - 1j*np.imag(denom)/d_mag_sq)
        elif t_type == 2:
            B = stack.pop()
            A = stack.pop()
            with np.errstate(invalid='ignore', over='ignore'):
                if op == OP_ADD: stack.append(A + B)
                elif op == OP_SUB: stack.append(A - B)
                elif op == OP_MUL: stack.append(A * B)
                elif op == OP_DIV:
                    denom = np.real(B)**2 + np.imag(B)**2 + EPS_REG
                    stack.append((np.real(A)*np.real(B) + np.imag(A)*np.imag(B))/denom + 1j*(np.imag(A)*np.real(B) - np.real(A)*np.imag(B))/denom)
                elif op == OP_POW:
                    mag_sq = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    ln_A = 0.5 * np.log(mag_sq) + 1j * np.arctan2(np.imag(A), np.real(A))
                    prod = B * ln_A
                    stack.append(np.exp(np.clip(np.real(prod), -15.0, 15.0) + 1j * np.imag(prod)))
                    
    Z_next = stack[0]
    anomalies = np.isnan(Z_next) | np.isinf(Z_next)
    if np.any(anomalies):
        Z_next = np.where(anomalies, complex_dtype(1e5 + 0j), Z_next)
    return Z_next


# --- Итераторы сеток (с проверкой дедлайна) ---
def compute_procedural_grid_pytorch(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, device, use_double=False, deadline=None):
    torch_dtype = torch.float64 if use_double else torch.float32
    torch_complex = torch.complex128 if use_double else torch.complex64

    x = torch.linspace(xmin, xmax, width, dtype=torch_dtype, device=device)
    y = torch.linspace(ymin, ymax, height, dtype=torch_dtype, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    C = torch.complex(X, Y)
    
    if is_julia:
        Z = C.clone()
        C_param = torch.tensor(c, dtype=torch_complex, device=device)
    else:
        Z = torch.zeros_like(C)
        C_param = C
        
    img = torch.zeros(C.shape, dtype=torch_dtype, device=device)
    mask = torch.ones(C.shape, dtype=torch.bool, device=device)
    
    Z_prev, Z_prev2 = torch.zeros_like(C), torch.zeros_like(C)
    R_esc_sq, eps_att_sq = 1e8, 1e-12
    
    with torch.no_grad():
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                raise TimeoutError("Превышен жесткий лимит времени вычислений.")
                
            Z_next = evaluate_rpn_pytorch(rpn, Z, C_param, device, use_double)
            mag_sq = Z_next.real**2 + Z_next.imag**2
            escaped = mag_sq > R_esc_sq
            
            dist_prev_sq = (Z_next.real - Z_prev.real)**2 + (Z_next.imag - Z_prev.imag)**2
            dist_prev2_sq = (Z_next.real - Z_prev2.real)**2 + (Z_next.imag - Z_prev2.imag)**2
            attracted = (dist_prev_sq < eps_att_sq) | (dist_prev2_sq < eps_att_sq)
            
            finished = escaped | attracted
            newly_finished = finished & mask
            
            if torch.any(newly_finished):
                z_mag = torch.clamp(torch.sqrt(mag_sq[newly_finished]), min=1.001)
                z_prev_mag = torch.clamp(torch.sqrt(Z.real**2 + Z.imag**2)[newly_finished], min=1.001)
                alpha = torch.clamp(torch.log(z_mag) / (torch.log(z_prev_mag) + 1e-20), min=1.1)
                nu = torch.log(torch.log(z_mag)) / torch.log(alpha)
                
                esc_subset = escaped[newly_finished]
                val = torch.where(esc_subset, i + 1.0 - nu, torch.tensor(float(i), dtype=torch_dtype, device=device))
                img[newly_finished] = val
                
            mask = mask & ~finished
            if not torch.any(mask):
                break
                
            Z_prev2, Z_prev, Z = Z_prev.clone(), Z.clone(), Z_next
            
        img[mask] = max_iter
    return img.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy()

def compute_procedural_grid_numpy(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, use_double=False, deadline=None):
    dtype = np.float64 if use_double else np.float32
    complex_dtype = np.complex128 if use_double else np.complex64

    x = np.linspace(xmin, xmax, width, dtype=dtype)
    y = np.linspace(ymin, ymax, height, dtype=dtype)
    X, Y = np.meshgrid(x, y)
    C = (X + 1j * Y).astype(complex_dtype)
    
    Z = C.copy() if is_julia else np.zeros_like(C)
    C_param = np.array(c, dtype=complex_dtype) if is_julia else C
    
    img = np.zeros(C.shape, dtype=float)
    mask = np.ones(C.shape, dtype=bool)
    
    Z_prev, Z_prev2 = np.zeros_like(C), np.zeros_like(C)
    R_esc_sq, eps_att_sq = 1e8, 1e-12
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            raise TimeoutError("Превышен жесткий лимит времени вычислений.")
            
        Z_next = evaluate_rpn_numpy(rpn, Z, C_param, use_double)
        mag_sq = np.real(Z_next)**2 + np.imag(Z_next)**2
        escaped = mag_sq > R_esc_sq
        
        dist_prev_sq = np.real(Z_next - Z_prev)**2 + np.imag(Z_next - Z_prev)**2
        dist_prev2_sq = np.real(Z_next - Z_prev2)**2 + np.imag(Z_next - Z_prev2)**2
        attracted = (dist_prev_sq < eps_att_sq) | (dist_prev2_sq < eps_att_sq)
        
        finished = escaped | attracted
        newly_finished = finished & mask
        
        if np.any(newly_finished):
            z_mag = np.maximum(np.sqrt(mag_sq[newly_finished]), 1.001)
            prev_mag = np.maximum(np.sqrt(np.real(Z)**2 + np.imag(Z)**2)[newly_finished], 1.001)
            alpha = np.maximum(np.log(z_mag) / (np.log(prev_mag) + 1e-20), 1.1)
            nu = np.log(np.log(z_mag)) / np.log(alpha)
            
            img[newly_finished] = np.where(escaped[newly_finished], i + 1.0 - nu, float(i))
            
        mask = mask & ~finished
        if not np.any(mask):
            break
            
        Z_prev2, Z_prev, Z = Z_prev.copy(), Z.copy(), Z_next
        
    img[mask] = max_iter
    return img, x, y

def safe_compute_grid(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, use_double=False, deadline=None, force_cpu=False):
    if HAS_TORCH and DEVICE.type == 'cuda' and not force_cpu:
        try:
            return compute_procedural_grid_pytorch(
                xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, DEVICE, use_double, deadline
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                log("ERROR", "DEVICE", "Переполнение видеопамяти CUDA. Переход на CPU.")
    return compute_procedural_grid_numpy(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, use_double, deadline)

# --- Навигация и контроль качества ---
def find_boundary_point_v2(img, x, y, max_iter, rng):
    dy, dx = np.gradient(img)
    grad = np.sqrt(dx**2 + dy**2)
    grad_norm = grad / (np.max(grad) + 1e-8)
    
    body_mask = (img >= max_iter - 1.0).astype(float)
    m_bar = fast_uniform_filter(body_mask, size=15)
    phi = 1.0 - 2.0 * np.abs(m_bar - 0.5)
    
    score_map = grad_norm * phi
    threshold = np.percentile(score_map, 90)
    indices = np.argwhere(score_map >= threshold)
    
    if len(indices) == 0: indices = np.argwhere(score_map > 0)
    if len(indices) == 0: return x[len(x)//2], y[len(y)//2]
    
    idx = indices[rng.randint(0, len(indices) - 1)]
    return x[idx[1]], y[idx[0]]

def find_highly_decorated_c_v2(rpn, rng, deadline=None):
    xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
    for _ in range(5):  
        img, x, y = safe_compute_grid(xmin, xmax, ymin, ymax, 150, 150, 100, rpn, False, 0j, use_double=False, deadline=deadline)
        target_x, target_y = find_boundary_point_v2(img, x, y, 100, rng)
        range_x, range_y = (xmax - xmin)/2.5, (ymax - ymin)/2.5
        xmin, xmax = target_x - range_x/2, target_x + range_x/2
        ymin, ymax = target_y - range_y/2, target_y + range_y/2
    return complex(target_x, target_y)

def apply_adaptive_tonemapping(img, max_iter):
    img_corrected = np.copy(img)
    body_mask = img >= max_iter - 1.0
    non_body_mask = ~body_mask
    
    if not np.any(non_body_mask): return None
    
    non_body_vals = img_corrected[non_body_mask]
    v_min, v_max = np.min(non_body_vals), np.max(non_body_vals)
    v_norm = (non_body_vals - v_min)/(v_max - v_min + 1e-10)
    
    gamma_dyn = np.clip(0.45 * ((np.median(v_norm) + 1e-5)/0.5)**0.65, 0.15, 0.45)
    
    v_final = 0.7 * np.power(v_norm, gamma_dyn) + 0.3 * np.log1p(15.0 * v_norm)/np.log1p(15.0)
    img_corrected[non_body_mask] = v_final
    img_corrected[body_mask] = np.nan
    return img_corrected

def check_aesthetic_quality(processed_img):
    if processed_img is None: return False
    body_mask = np.isnan(processed_img)
    body_ratio = np.sum(body_mask) / body_mask.size
    
    if not (0.015 < body_ratio < 0.65): return False
    
    non_body = processed_img[~body_mask]
    if non_body.size < 200: return False
    
    std_val = np.std(non_body)
    if std_val < 0.12: return False
    
    unique_vals = np.unique(np.round(non_body, 2))
    if len(unique_vals) < 20: return False
    return True


# --- Высококачественный пиксель-в-пиксель экспорт напрямую через PIL ---
def export_to_buffers_pil(processed_img, cmap=None, target_res=1600):
    if cmap is None:
        cmap = CLASSIC_CMAP
        
    body_mask = np.isnan(processed_img)
    clean_img = np.nan_to_num(processed_img, nan=0.0)
    
    rgba_img = cmap(clean_img)
    rgba_img[body_mask] = [0.0, 0.0, 0.0, 1.0] # Тело фрактала красим в черный
    
    rgb_img = (rgba_img[:, :, :3] * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(rgb_img)
    
    # SSAA Даунсамплинг (Lanczos)
    if img_pil.width > target_res:
        img_pil = img_pil.resize((target_res, target_res), Image.Resampling.LANCZOS)
        
    # Экспорт JPEG
    buf_jpeg = io.BytesIO()
    img_pil.save(buf_jpeg, format='JPEG', quality=90, optimize=True)
    buf_jpeg.seek(0)
    
    # Экспорт оригинального PNG без потерь
    buf_png = io.BytesIO()
    img_pil.save(buf_png, format='PNG')
    buf_png.seek(0)
    
    return buf_jpeg, buf_png


# --- Оптимизированный генератор фракталов с лимитом времени 2 минуты ---
def generate_fractal_pipeline(quality_res=1600, steps=10, progress_callback=None, force_cpu=False):
    start_time = time.time()
    deadline = start_time + 120.0  # Жесткий лимит 120 секунд (2 минуты)

    rng = random.Random(secrets.randbits(128))
    seed_int = rng.randint(0, 2**128 - 1)
    
    decoder = EntropyDecoder(seed_int)
    ast_tree = generate_ast(decoder, depth=1, max_depth=4) 
    rpn_tokens = []
    ast_to_rpn(ast_tree, rpn_tokens)
    
    while not validate_rpn(rpn_tokens):
        if time.time() > deadline:
            raise TimeoutError("Таймаут безопасности превышен на этапе генерации RPN.")
        seed_int = rng.randint(0, 2**128 - 1)
        decoder = EntropyDecoder(seed_int)
        ast_tree = generate_ast(decoder, depth=1, max_depth=4)
        rpn_tokens = []
        ast_to_rpn(ast_tree, rpn_tokens)
        
    formula_str = rpn_to_str(rpn_tokens)
    is_julia = (rng.randint(0, 1) == 1)
    
    if progress_callback:
        progress_callback("🌀 Расчёт векторов комплексного поля...")
    c_val = find_highly_decorated_c_v2(rpn_tokens, rng, deadline=deadline) if is_julia else 0j
    
    xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
    for step in range(1, steps + 1):
        if progress_callback:
            progress_callback(f"🪐 Позиционирование зума (Шаг {step}/{steps})...")
        current_max_iter = 120 + step * 60
        img, x, y = safe_compute_grid(
            xmin, xmax, ymin, ymax, 250, 250, current_max_iter, rpn_tokens, is_julia, c_val, use_double=False, deadline=deadline, force_cpu=force_cpu
        )
        target_x, target_y = find_boundary_point_v2(img, x, y, current_max_iter, rng)
        range_x, range_y = (xmax - xmin)/2.5, (ymax - ymin)/2.5
        xmin, xmax = target_x - range_x/2, target_x + range_x/2
        ymin, ymax = target_y - range_y/2, target_y + range_y/2
        
    final_max_iter = 500
    
    # --- БЫСТРЫЙ ПРЕВЬЮ ПАСС: Ранняя отбраковка ---
    if progress_callback:
        progress_callback("⚡ Проверка эстетического потенциала...")
    
    preview_res = 200
    preview_img, _, _ = safe_compute_grid(
        xmin, xmax, ymin, ymax, preview_res, preview_res, final_max_iter, rpn_tokens, is_julia, c_val, use_double=False, deadline=deadline, force_cpu=force_cpu
    )
    preview_processed = apply_adaptive_tonemapping(preview_img, final_max_iter)
    
    if not check_aesthetic_quality(preview_processed):
        log("WARN", "QUALITY", "Фрактал отклонён на стадии быстрого превью.")
        return None, None, None, None
        
    # --- ФИНАЛЬНЫЙ РЕНДЕР (Адаптивные параметры качества) ---
    if HAS_TORCH and DEVICE.type == 'cuda':
        target_res = quality_res 
        ssaa_factor = 1.5       
    else:
        target_res = 1200       
        ssaa_factor = 1.0       
        
    render_res = int(target_res * ssaa_factor)
    
    log("COMPUTE", "GRID", f"Начат рендеринг высокого разрешения {render_res}x{render_res}...")
    if progress_callback:
        progress_callback(f"🧬 Рендеринг фрактала высокой точности ({target_res}x{target_res})...")
        
    final_img, _, _ = safe_compute_grid(
        xmin, xmax, ymin, ymax, render_res, render_res, final_max_iter, rpn_tokens, is_julia, c_val, use_double=True, deadline=deadline, force_cpu=force_cpu
    )
    
    processed_img = apply_adaptive_tonemapping(final_img, final_max_iter)
    
    if not check_aesthetic_quality(processed_img):
        log("WARN", "QUALITY", "Фрактал отклонён финальным фильтром эстетического качества.")
        return None, None, None, None
        
    buf_jpeg, buf_png = export_to_buffers_pil(processed_img, CLASSIC_CMAP, target_res=target_res)
    coords_dict = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    
    return buf_jpeg, buf_png, formula_str, coords_dict


# --- Умный планировщик автоматической рассылки подписчикам ---
BROADCAST_STATE_FILE = "broadcast_state.json"
broadcast_lock = threading.Lock()

def load_broadcast_state():
    with broadcast_lock:
        if not os.path.exists(BROADCAST_STATE_FILE):
            return {"last_broadcast_epoch": 0.0}
        try:
            with open(BROADCAST_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"last_broadcast_epoch": 0.0}

def save_broadcast_state(state):
    with broadcast_lock:
        try:
            with open(BROADCAST_STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            log("ERROR", "STORAGE", f"Ошибка сохранения состояния рассылки: {e}")

def run_broadcast_distribution():
    subs = load_subscribers()
    if not subs:
        log("INFO", "AUTO", "Подписчиков для рассылки нет.")
        return
        
    log("INFO", "AUTO", f"Инициация рассылки для {len(subs)} пользователей...")
    buf_jpeg, buf_png, formula, coords = None, None, None, None
    for _ in range(15):  
        try:
            buf_jpeg, buf_png, formula, coords = generate_fractal_pipeline(quality_res=1600, steps=10, force_cpu=True)
            if buf_jpeg is not None:
                break
        except TimeoutError:
            log("WARN", "AUTO", "Прервано по таймауту. Поиск новой сингулярности...")
            
    if buf_jpeg is not None:
        for chat_id in list(subs):
            try:
                buf_jpeg.seek(0)
                buf_png.seek(0)
                
                # Динамически получаем язык каждого конкретного подписчика
                user_lang = get_user_setting(chat_id, "lang", None)
                if not user_lang:
                    user_lang = "en" # По умолчанию для рассылки, если язык не определен
                    
                t = TRANSLATIONS[user_lang]
                
                safe_send_photo(
                    chat_id,
                    buf_jpeg,
                    caption=t["broadcast_caption"].format(
                        formula=formula,
                        xmin=coords['xmin'], xmax=coords['xmax'],
                        ymin=coords['ymin'], ymax=coords['ymax']
                    ),
                    parse_mode='Markdown'
                )
                
                buf_png.name = f"fractal_{secrets.token_hex(4)}.png"
                safe_send_document(
                    chat_id,
                    buf_png,
                    caption="🖼️ **PNG**" if user_lang == "en" else "🖼️ **PNG-оригинал без сжатия**",
                    parse_mode='Markdown'
                )
                log("SUCCESS", "AUTO", f"Фрактал доставлен подписчику {chat_id} [Язык: {user_lang}].")
            except telebot.apihelper.ApiTelegramException as e:
                if e.error_code in [403, 400]:
                    log("WARN", "AUTO", f"Пользователь {chat_id} заблокировал бота. Удаление подписки.")
                    remove_subscriber(chat_id)
                    stats.set_user_inactive(chat_id)
            except Exception as e:
                log("ERROR", "AUTO", f"Ошибка отправки пользователю {chat_id}: {e}")
            
            time.sleep(0.25)
            
        buf_jpeg.close()
        buf_png.close()

def automated_delivery_loop():
    INTERVAL = 7200 # 2 часа в секундах
    while True:
        try:
            now = time.time()
            state = load_broadcast_state()
            last_sent = state.get("last_broadcast_epoch", 0.0)
            
            # Находим время текущего планового интервала
            current_scheduled_slot = (now // INTERVAL) * INTERVAL
            
            # Инициализация при первом запуске
            if last_sent == 0.0:
                state["last_broadcast_epoch"] = current_scheduled_slot
                save_broadcast_state(state)
                last_sent = current_scheduled_slot
            
            # Если пропущен последний слот (даже если бот пропустил 2 и более интервалов)
            if now >= current_scheduled_slot and last_sent < current_scheduled_slot:
                log("INFO", "AUTO", f"Запуск рассылки за слот {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_scheduled_slot))}")
                
                run_broadcast_distribution()
                
                # Записываем завершение рассылки. Прошлые пропуски игнорируются для защиты от лавины отправки
                state["last_broadcast_epoch"] = current_scheduled_slot
                save_broadcast_state(state)
                
        except Exception as e:
            log("ERROR", "AUTO", f"Критическая ошибка планировщика рассылки: {e}")
            
        time.sleep(30)


# --- Динамический интерфейс ---
def get_main_keyboard(chat_id, lang="ru"):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    
    t = TRANSLATIONS[lang]
    btn_gen = types.KeyboardButton(t["btn_shallow"])
    btn_lucky = types.KeyboardButton(t["btn_deep"])
    
    subs = load_subscribers()
    if chat_id in subs:
        btn_sub = types.KeyboardButton(t["btn_sub_stop"])
    else:
        btn_sub = types.KeyboardButton(t["btn_sub_start"])
        
    btn_batch3 = types.KeyboardButton(t["btn_batch3"])
    btn_batch5 = types.KeyboardButton(t["btn_batch5"])
    btn_custom = types.KeyboardButton(t["btn_custom"])
    btn_settings = types.KeyboardButton(t["btn_settings"])
    btn_help = types.KeyboardButton(t["btn_help"])
    
    markup.row(btn_gen, btn_lucky)
    markup.row(btn_batch3, btn_batch5)
    markup.row(btn_sub)
    markup.row(btn_custom, btn_settings)
    markup.row(btn_help)
    return markup

HELP_TEXT_RU = (
    "👁‍⚙ **Фрактальный навигатор — справка**\n\n"
    "Бот генерирует уникальные процедурные фракталы на основе математических формул. "
    "Каждое изображение создаётся с нуля и проходит многоступенчатые эстетические тесты.\n\n"
    "🎛 **Кнопки управления:**\n\n"
    "🌌 *Малые расстояния* – стандартное погружение со случайным зумом (от 4 до 10 шагов). "
    "Позволяет увидеть общие очертания и гармонию фрактала.\n\n"
    "🌀 *Сверхглубокий зум* – глубокое погружение (от 15 до 30 шагов). "
    "Исследует микроскопические детали в глубине хаоса. Требует больше ресурсов.\n\n"
    "🧿 *Запустить бесконечный поток* – бот будет автоматически каждые 2 часа присылать вам новый фрактал.\n"
    "⏳ *Остановить поток* – отключает автоматическую рассылку.\n\n"
    "🔮 *Пакет из 3 / 5 фракталов* – последовательная генерация нескольких фракталов "
    "(глубина зума настраивается в меню настроек).\n\n"
    "✍️ *Свой фрактал* – позволяет вам ввести собственную формулу и диапазон координат для точного рендеринга.\n\n"
    "⚙️ *Настройки* – параметры масштабирования для пакетного рендеринга.\n\n"
    "❓ *Помощь* – показывает это сообщение.\n\n"
    "⚙️ **Технические детали:**\n"
    "• Каждая проекция выводит точные координаты и формулу, чтобы вы могли воспроизвести её позже!\n"
    "• Одно вычисление длится не более 120 секунд.\n"
    "• Между генерациями действует пауза 4 секунды.\n\n"
    "Приятных погружений! 🌀"
)

HELP_TEXT_EN = (
    "👁‍⚙ **Fractal Navigator — Help**\n\n"
    "The bot generates unique procedural fractals using mathematical formulas. "
    "Each image is rendered from scratch and passes multi-stage aesthetic evaluations.\n\n"
    "🎛 **Control Keys:**\n\n"
    "🌌 *Shallow Zoom* – standard dive with random zoom (4 to 10 steps). "
    "Allows observing the general shape and harmony of the fractal.\n\n"
    "🌀 *Deep Zoom* – deep dive (15 to 30 steps). "
    "Explores microscopic details deep inside the chaos. Requires more resources.\n\n"
    "🧿 *Start Infinite Stream* – the bot will automatically send you a new fractal every 2 hours.\n"
    "⏳ *Stop Infinite Stream* – disables automated delivery.\n\n"
    "🔮 *Batch of 3 / 5* – sequential generation of several fractals "
    "(zoom depth can be configured in settings).\n\n"
    "✍️ *Custom Fractal* – allows you to enter your own formula and coordinate range for exact rendering.\n\n"
    "⚙️ *Settings* – scaling parameters for batch rendering and language settings.\n\n"
    "❓ *Help* – shows this message.\n\n"
    "⚙️ **Technical Details:**\n"
    "• Each projection outputs precise coordinates and formula so you can replicate it later!\n"
    "• One computation is limited to 120 seconds.\n"
    "• There is a 4-second cooldown between manually requested generations.\n\n"
    "Enjoy the dive! 🌀"
)

HELP_TEXT_EN = (
    "👁‍⚙ **Fractal Navigator — Help**\n\n"
    "The bot generates unique procedural fractals using mathematical formulas. "
    "Each image is rendered from scratch and passes multi-stage aesthetic evaluations.\n\n"
    "🎛 **Control Keys:**\n\n"
    "🌌 *Shallow Zoom* – standard dive with random zoom (4 to 10 steps). "
    "Allows observing the general shape and harmony of the fractal.\n\n"
    "🌀 *Deep Zoom* – deep dive (15 to 30 steps). "
    "Explores microscopic details deep inside the chaos. Requires more resources.\n\n"
    "🧿 *Start Infinite Stream* – the bot will automatically send you a new fractal every 2 hours.\n"
    "⏳ *Stop Infinite Stream* – disables automated delivery.\n\n"
    "🔮 *Batch of 3 / 5* – sequential generation of several fractals "
    "(zoom depth can be configured in settings).\n\n"
    "✍️ *Custom Fractal* – allows you to enter your own formula and coordinate range for exact rendering.\n\n"
    "⚙️ *Settings* – scaling parameters for batch rendering and language settings.\n\n"
    "❓ *Help* – shows this message.\n\n"
    "⚙️ **Technical Details:**\n"
    "• Each projection outputs precise coordinates and formula so you can replicate it later!\n"
    "• One computation is limited to 120 seconds.\n"
    "• There is a 4-second cooldown between manually requested generations.\n\n"
    "Enjoy the dive! 🌀"
)

# --- Система отправки пингов на облако (Наблюдатель) ---
def heartbeat_loop():
    while True:
        try:
            requests.get("https://MHPFan.pythonanywhere.com/ping", timeout=10)
        except Exception as e:
            log("WARN", "SYSTEM", f"Не удалось отправить пинг на сервер-наблюдатель: {e}")
        time.sleep(60)


# --- Telegram Bot Handlers ---
@bot.message_handler(commands=['start', 'help', 'restart'])
def send_welcome(message):
    try:
        chat_id = message.chat.id
        stats.register_user(chat_id)
        lang = get_user_lang(message)
        
        safe_api_call(
            bot.send_message,
            chat_id, 
            TRANSLATIONS[lang]["welcome"],
            reply_markup=get_main_keyboard(chat_id, lang),
            parse_mode='Markdown'
        )
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Ошибка отправки приветствия: {e}")

@bot.message_handler(func=lambda message: message.text in ["❓ Помощь", "❓ Help"])
def send_help(message):
    try:
        chat_id = message.chat.id
        stats.register_user(chat_id)
        lang = get_user_lang(message)
        
        safe_api_call(
            bot.send_message,
            chat_id,
            TRANSLATIONS[lang]["help"],
            reply_markup=get_main_keyboard(chat_id, lang),
            parse_mode='Markdown'
        )
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Ошибка отправки справки: {e}")

@bot.message_handler(func=lambda message: message.text in ["⚙️ Настройки", "⚙️ Settings"])
def show_settings(message):
    chat_id = message.chat.id
    stats.register_user(chat_id)
    lang = get_user_lang(message)
    t = TRANSLATIONS[lang]
    
    mode = get_user_setting(chat_id, "zoom_mode", "shallow")
    if lang == "ru":
        mode_text = "🌌 Малые расстояния" if mode == "shallow" else "🌀 Сверхглубокий зум"
    else:
        mode_text = "🌌 Shallow Zoom" if mode == "shallow" else "🌀 Deep Zoom"
    
    markup = types.InlineKeyboardMarkup()
    btn_toggle = types.InlineKeyboardButton(t["btn_toggle_zoom"], callback_data="toggle_zoom_mode")
    btn_lang = types.InlineKeyboardButton(t["btn_toggle_lang"], callback_data="toggle_lang")
    markup.add(btn_toggle)
    markup.add(btn_lang)
    
    safe_api_call(
        bot.send_message,
        chat_id,
        f"{t['settings_title']}\n\n"
        f"{t['settings_desc'].format(mode=mode_text)}\n\n"
        f"{t['settings_lang']}",
        reply_markup=markup,
        parse_mode='Markdown'
    )

@bot.callback_query_handler(func=lambda call: call.data in ["toggle_zoom_mode", "toggle_lang"])
def callback_settings(call):
    chat_id = call.message.chat.id
    lang = get_user_lang(call.message)
    
    if call.data == "toggle_zoom_mode":
        current_mode = get_user_setting(chat_id, "zoom_mode", "shallow")
        new_mode = "deep" if current_mode == "shallow" else "shallow"
        save_user_setting(chat_id, "zoom_mode", new_mode)
        
    elif call.data == "toggle_lang":
        new_lang = "en" if lang == "ru" else "ru"
        save_user_setting(chat_id, "lang", new_lang)
        lang = new_lang  # Обновляем локальную переменную для вывода нового текста
        
    t = TRANSLATIONS[lang]
    mode = get_user_setting(chat_id, "zoom_mode", "shallow")
    if lang == "ru":
        mode_text = "🌌 Малые расстояния" if mode == "shallow" else "🌀 Сверхглубокий зум"
    else:
        mode_text = "🌌 Shallow Zoom" if mode == "shallow" else "🌀 Deep Zoom"
        
    markup = types.InlineKeyboardMarkup()
    btn_toggle = types.InlineKeyboardButton(t["btn_toggle_zoom"], callback_data="toggle_zoom_mode")
    btn_lang = types.InlineKeyboardButton(t["btn_toggle_lang"], callback_data="toggle_lang")
    markup.add(btn_toggle)
    markup.add(btn_lang)
    
    try:
        safe_edit_message_text(
            f"{t['settings_title']}\n\n"
            f"{t['settings_desc'].format(mode=mode_text)}\n\n"
            f"{t['settings_lang']}",
            chat_id,
            call.message.message_id,
            reply_markup=markup,
            parse_mode='Markdown'
        )
        # Также обновляем Reply-кнопки меню на лету под новый язык!
        safe_api_call(
            bot.send_message,
            chat_id,
            t["lang_changed"] if call.data == "toggle_lang" else "✅",
            reply_markup=get_main_keyboard(chat_id, lang)
        )
        bot.answer_callback_query(call.id)
    except Exception:
        pass

@bot.message_handler(func=lambda message: message.text in ["✍️ Свой фрактал", "✍️ Custom Fractal"])
def request_custom_fractal(message):
    chat_id = message.chat.id
    stats.register_user(chat_id)
    lang = get_user_lang(message)
    t = TRANSLATIONS[lang]
    
    safe_api_call(
        bot.send_message,
        chat_id,
        t["custom_info"],
        parse_mode='Markdown',
        reply_markup=types.ForceReply(selective=True)
    )

@bot.message_handler(func=lambda msg: msg.reply_to_message and (
    "Пришлите ваши параметры ответным сообщением" in msg.reply_to_message.text or
    "Send your parameters in a reply" in msg.reply_to_message.text
))
def handle_custom_fractal_input(message):
    chat_id = message.chat.id
    text = message.text
    lang = get_user_lang(message)
    t = TRANSLATIONS[lang]
    
    formula_str = None
    coords_tuple = None
    
    # Парсинг полей независимо от языка
    lines = text.split("\n")
    for line in lines:
        if "Формула:" in line or "Formula:" in line:
            # Поддержка обоих вариантов написания
            parts = line.split("Формула:") if "Формула:" in line else line.split("Formula:")
            formula_str = parts[1].strip()
        elif "Координаты:" in line or "Coordinates:" in line:
            parts = line.split("Координаты:") if "Координаты:" in line else line.split("Coordinates:")
            coords_str = parts[1].strip()
            try:
                coords_tuple = tuple(map(float, coords_str.replace(" ", "").split(",")))
            except Exception:
                pass
                
    if not formula_str:
        formula_str = text.strip()
    
    if coords_tuple and len(coords_tuple) != 4:
        coords_tuple = None
        
    try:
        rpn_tokens = parse_infix_to_rpn(formula_str)
        if not validate_rpn(rpn_tokens):
            raise ValueError("Формула должна содержать как минимум переменные Z и C, а также оператор.")
    except Exception as e:
        safe_api_call(
            bot.send_message,
            chat_id,
            t["custom_error"].format(error=str(e)),
            parse_mode='Markdown'
        )
        return
        
    status, val = user_manager.try_start_job(chat_id)
    if status == "busy":
        safe_api_call(bot.send_message, chat_id, t["busy"])
        return
    elif status == "cooldown":
        safe_api_call(bot.send_message, chat_id, t["cooldown"].format(val=val))
        return
        
    try:
        status_msg = safe_api_call(bot.send_message, chat_id, "🧬 ...")
        updater = ProgressUpdater(bot, chat_id, status_msg.message_id)
        
        if coords_tuple:
            xmin, xmax, ymin, ymax = coords_tuple
            steps = 0 
        else:
            xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
            steps = 6 
            
        is_julia = False 
        c_val = 0j
        
        if steps > 0:
            rng = random.Random(secrets.randbits(128))
            updater.update(f"🧬 {t['zoom_position'].format(step=1, steps=steps)}")
            for step in range(1, steps + 1):
                current_max_iter = 120 + step * 60
                img, x, y = safe_compute_grid(
                    xmin, xmax, ymin, ymax, 250, 250, current_max_iter, rpn_tokens, is_julia, c_val, use_double=False
                )
                target_x, target_y = find_boundary_point_v2(img, x, y, current_max_iter, rng)
                range_x, range_y = (xmax - xmin)/2.5, (ymax - ymin)/2.5
                xmin, xmax = target_x - range_x/2, target_x + range_x/2
                ymin, ymax = target_y - range_y/2, target_y + range_y/2
        
        final_max_iter = 500
        target_res = 1600 if (HAS_TORCH and DEVICE.type == 'cuda') else 1200
        ssaa_factor = 1.5 if (HAS_TORCH and DEVICE.type == 'cuda') else 1.0
        render_res = int(target_res * ssaa_factor)
        
        updater.update(t["rendering_high"].format(res=target_res))
        
        final_img, _, _ = safe_compute_grid(
            xmin, xmax, ymin, ymax, render_res, render_res, final_max_iter, rpn_tokens, is_julia, c_val, use_double=True
        )
        
        processed_img = apply_adaptive_tonemapping(final_img, final_max_iter)
        if processed_img is None:
            processed_img = np.nan_to_num(final_img)
            
        buf_jpeg, buf_png = export_to_buffers_pil(processed_img, CLASSIC_CMAP, target_res=target_res)
        
        try:
            bot.delete_message(chat_id, status_msg.message_id)
        except Exception:
            pass
        
        safe_send_photo(
            chat_id, 
            buf_jpeg, 
            caption=t["custom_success"].format(
                formula=formula_str,
                xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax
            ), 
            parse_mode='Markdown',
            reply_markup=get_main_keyboard(chat_id, lang)
        )
        
        buf_png.name = f"custom_fractal_{secrets.token_hex(4)}.png"
        safe_send_document(
            chat_id,
            buf_png,
            caption=t["png_caption"],
            parse_mode='Markdown'
        )
        
        buf_jpeg.close()
        buf_png.close()
        
        stats.log_generation(chat_id, "custom", final_max_iter)
        
    except Exception as e:
        log("ERROR", "CUSTOM_RENDER", f"Ошибка при кастомном рендере: {e}")
        safe_api_call(bot.send_message, chat_id, t["gen_error"].format(error=str(e)), reply_markup=get_main_keyboard(chat_id, lang))
    finally:
        user_manager.end_job(chat_id)

@bot.message_handler(func=lambda message: message.text in ["🌌 Малые расстояния", "🌌 Shallow Zoom"])
@bot.message_handler(func=lambda message: message.text in ["🌀 Сверхглубокий зум", "🌀 Deep Zoom"])
@bot.message_handler(commands=['generate'])
def send_fractal(message):
    chat_id = message.chat.id
    username = message.from_user.username or "NoUsername"
    lang = get_user_lang(message)
    t = TRANSLATIONS[lang]

    log("INFO", "TELEGRAM", f"Пользователь {chat_id} (@{username}) запросил генерацию: '{message.text}' [Язык: {lang}]")

    if "Сверхглубокий" in message.text or "Deep" in message.text:
        steps_min, steps_max = 15, 30
        mode_text = t["status_deep"]
        gen_type = "deep"
    else:
        steps_min, steps_max = 4, 10
        mode_text = t["status_shallow"]
        gen_type = "shallow"

    status, val = user_manager.try_start_job(chat_id)
    if status == "busy":
        safe_api_call(bot.send_message, chat_id, t["busy"])
        return
    elif status == "cooldown":
        safe_api_call(bot.send_message, chat_id, t["cooldown"].format(val=val))
        return

    try:
        random_steps = random.randint(steps_min, steps_max)
        status_msg = safe_api_call(bot.send_message, chat_id, mode_text.format(steps=random_steps))
        updater = ProgressUpdater(bot, chat_id, status_msg.message_id)
        
        buf_jpeg, buf_png, formula, coords = None, None, None, None
        max_attempts = 15
        
        for attempt in range(1, max_attempts + 1):
            def make_callback(att):
                return lambda text: updater.update(f"{t['attempt'].format(attempt=att, max_attempts=max_attempts)}\n└ {text}")
            
            updater.update(t["init_grid"].format(attempt=attempt, max_attempts=max_attempts), force=True)
            
            try:
                # Передаем переводы прогресса внутрь пайплайна через замену коллбэков (для простоты оставляем общие вызовы, локализуя текст)
                localized_cb = lambda step_txt: make_callback(attempt)(
                    t["vectors"] if "векторов" in step_txt else
                    t["zoom_position"].format(step=re.findall(r'\d+', step_txt)[0], steps=random_steps) if "Позиционирование" in step_txt else
                    t["check_aesthetic"] if "эстетического" in step_txt else
                    t["rendering_high"].format(res=1600 if (HAS_TORCH and DEVICE.type == 'cuda') else 1200) if "Рендеринг" in step_txt else step_txt
                )
                
                buf_jpeg, buf_png, formula, coords = generate_fractal_pipeline(
                    quality_res=1600, 
                    steps=random_steps, 
                    progress_callback=localized_cb
                )
            except TimeoutError as te:
                log("ERROR", "TIMEOUT", f"Таймаут вычислений на попытке {attempt}: {te}")
                if attempt == max_attempts:
                    raise te
                updater.update(t["timeout_retry"].format(attempt=attempt, max_attempts=max_attempts), force=True)
                time.sleep(1.0)
                continue
            
            if buf_jpeg is not None:
                break
            else:
                updater.update(t["rejected_attempt"].format(attempt=attempt, max_attempts=max_attempts), force=True)
                time.sleep(1.0)
        
        if buf_jpeg is None:
            updater.update(t["unstable_chaos"], force=True)
            return
            
        try:
            bot.delete_message(chat_id, status_msg.message_id)
        except Exception:
            pass
        
        log("UPLOAD", "TELEGRAM", f"Отправка превью-фото {chat_id}...")
        
        phrase = random.choice(DEFAULT_PHRASES_EN) if lang == "en" else get_random_phrase()
        
        safe_send_photo(
            chat_id, 
            buf_jpeg, 
            caption=t["fractal_ready"].format(
                formula=formula,
                xmin=coords['xmin'], xmax=coords['xmax'],
                ymin=coords['ymin'], ymax=coords['ymax'],
                phrase=phrase
            ), 
            parse_mode='Markdown',
            reply_markup=get_main_keyboard(chat_id, lang)
        )
        
        log("UPLOAD", "TELEGRAM", f"Отправка оригинального PNG-файла {chat_id}...")
        buf_png.name = f"fractal_{secrets.token_hex(4)}.png"
        safe_send_document(
            chat_id,
            buf_png,
            caption=t["png_caption"],
            parse_mode='Markdown'
        )
        
        buf_jpeg.close()
        buf_png.close()
        stats.log_generation(chat_id, gen_type, random_steps)
        
    except TimeoutError:
        log("ERROR", "TELEGRAM", f"Генерация для {chat_id} полностью остановлена из-за превышения лимита времени.")
        try:
            safe_api_call(
                bot.send_message, 
                chat_id, 
                t["timeout_error"],
                reply_markup=get_main_keyboard(chat_id, lang)
            )
        except Exception:
            pass
    except telebot.apihelper.ApiTelegramException as te:
        if te.error_code in [403, 400]:
            log("WARN", "TELEGRAM", f"Пользователь {chat_id} заблокировал бота.")
            remove_subscriber(chat_id)
            stats.set_user_inactive(chat_id)
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Сбой отправки: {e}")
        try:
            safe_api_call(bot.send_message, chat_id, t["gen_error"].format(error=str(e)), reply_markup=get_main_keyboard(chat_id, lang))
        except Exception:
            pass
    finally:
        user_manager.end_job(chat_id)

@bot.message_handler(func=lambda message: message.text in [
    "🧿 Запустить бесконечный поток", "🧿 Start Infinite Stream",
    "⏳ Остановить бесконечный поток", "⏳ Stop Infinite Stream"
])
@bot.message_handler(commands=['subscribe', 'unsubscribe'])
def toggle_subscription(message):
    chat_id = message.chat.id
    lang = get_user_lang(message)
    t = TRANSLATIONS[lang]
    
    try:
        if "Запустить" in message.text or "Start" in message.text or message.text == "/subscribe":
            save_subscriber(chat_id)
            stats.log_subscription(chat_id, "subscribe")
            safe_api_call(
                bot.send_message,
                chat_id, 
                t["sub_started"],
                reply_markup=get_main_keyboard(chat_id, lang)
            )
        else:
            remove_subscriber(chat_id)
            stats.log_subscription(chat_id, "unsubscribe")
            safe_api_call(
                bot.send_message, 
                chat_id, 
                t["sub_stopped"],
                reply_markup=get_main_keyboard(chat_id, lang)
            )
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Ошибка изменения подписки {chat_id}: {e}")

@bot.message_handler(func=lambda message: message.text.startswith("🔮 Сгенерировать пакет") or message.text.startswith("🔮 Generate Batch"))
def send_batch_fractal(message):
    chat_id = message.chat.id
    username = message.from_user.username or "NoUsername"
    lang = get_user_lang(message)
    t = TRANSLATIONS[lang]
    
    num = 5 if "5" in message.text else 3
    
    status, val = user_manager.try_start_job(chat_id)
    if status == "busy":
        try:
            safe_api_call(bot.send_message, chat_id, t["busy"])
        except Exception:
            pass
        return
    elif status == "cooldown":
        try:
            safe_api_call(bot.send_message, chat_id, t["cooldown"].format(val=val))
        except Exception:
            pass
        return

    try:
        zoom_mode = get_user_setting(chat_id, "zoom_mode", "shallow")
        if zoom_mode == "deep":
            steps_min, steps_max = 15, 30
            mode_desc = "Сверхглубокий зум" if lang == "ru" else "Deep Zoom"
        else:
            steps_min, steps_max = 4, 10
            mode_desc = "Малые расстояния" if lang == "ru" else "Shallow Zoom"
            
        status_msg = safe_api_call(
            bot.send_message, 
            chat_id, 
            t["batch_init"].format(num=num, mode=mode_desc),
            parse_mode='Markdown'
        )
        log("INFO", "TELEGRAM", f"Пользователь {chat_id} (@{username}) запросил пакет из {num} фракталов (режим: {zoom_mode}).")
        
        updater = ProgressUpdater(bot, chat_id, status_msg.message_id)
        generated = 0
        
        for i in range(num):
            buf_jpeg, buf_png, formula, coords = None, None, None, None
            max_attempts = 15
            
            for attempt in range(1, max_attempts + 1):
                random_steps = random.randint(steps_min, steps_max)
                
                # Локализуем callback прогресс-бара внутри пакета
                def make_batch_callback(index, att):
                    return lambda text: updater.update(
                        t["batch_step"].format(
                            index=index+1, 
                            num=num, 
                            attempt=att, 
                            max_attempts=max_attempts, 
                            text=(
                                t["vectors"] if "векторов" in text else
                                t["zoom_position"].format(step=re.findall(r'\d+', text)[0], steps=random_steps) if "Позиционирование" in text else
                                t["check_aesthetic"] if "эстетического" in text else
                                t["rendering_high"].format(res=1600 if (HAS_TORCH and DEVICE.type == 'cuda') else 1200) if "Рендеринг" in text else text
                            )
                        ),
                        force=True
                    )
                
                updater.update(
                    t["batch_step"].format(
                        index=i+1, 
                        num=num, 
                        attempt=attempt, 
                        max_attempts=max_attempts, 
                        text=t["vectors"]
                    ),
                    force=True
                )
                
                try:
                    buf_jpeg, buf_png, formula, coords = generate_fractal_pipeline(
                        quality_res=1600, 
                        steps=random_steps, 
                        progress_callback=make_batch_callback(i, attempt)
                    )
                except TimeoutError:
                    log("WARN", "TIMEOUT", f"Пакет: Фрактал {i+1} на попытке {attempt} прерван по таймауту.")
                    if attempt == max_attempts:
                        break
                    updater.update(t["timeout_retry"].format(attempt=attempt, max_attempts=max_attempts), force=True)
                    time.sleep(1.0)
                    continue
                
                if buf_jpeg is not None:
                    break
                else:
                    updater.update(
                        t["batch_step"].format(
                            index=i+1, 
                            num=num, 
                            attempt=attempt, 
                            max_attempts=max_attempts, 
                            text=t["unstable_chaos"]
                        ),
                        force=True
                    )
                    time.sleep(1.0)
            
            if buf_jpeg is not None:
                try:
                    log("UPLOAD", "TELEGRAM", f"Отправка кадра {i+1}/{num} пользователю {chat_id}...")
                    safe_send_photo(
                        chat_id,
                        buf_jpeg,
                        caption=t["batch_ready"].format(
                            index=i+1,
                            formula=formula,
                            xmin=coords['xmin'], xmax=coords['xmax'],
                            ymin=coords['ymin'], ymax=coords['ymax']
                        ),
                        parse_mode='Markdown'
                    )
                    log("SUCCESS", "TELEGRAM", f"Кадр {i+1}/{num} успешно доставлен.")
                    generated += 1
                except telebot.apihelper.ApiTelegramException as te:
                    if te.error_code in [403, 400]:
                        log("WARN", "TELEGRAM", f"Пользователь {chat_id} заблокировал бота. Генерация пакета прервана.")
                        remove_subscriber(chat_id)
                        stats.set_user_inactive(chat_id)
                        break  
                except Exception as e:
                    log("ERROR", "TELEGRAM", f"Ошибка отправки кадра {i+1} пакета: {e}")
                finally:
                    buf_jpeg.close()
                    buf_png.close()
                time.sleep(1.0)
                
        try:
            bot.delete_message(chat_id, status_msg.message_id)
        except Exception:
            pass
            
        if generated == 0:
            safe_api_call(bot.send_message, chat_id, t["batch_failed"], reply_markup=get_main_keyboard(chat_id, lang))
        else:
            safe_api_call(bot.send_message, chat_id, t["batch_success"], reply_markup=get_main_keyboard(chat_id, lang))
            stats.log_generation(chat_id, f"batch_{zoom_mode}", num)
            
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Критическая ошибка пакетной генерации {chat_id}: {e}")
        try:
            safe_api_call(bot.send_message, chat_id, t["gen_error"].format(error=str(e)), reply_markup=get_main_keyboard(chat_id, lang))
        except Exception:
            pass
    finally:
        user_manager.end_job(chat_id)

# --- Админ-панель для визуализации аналитики ---
@bot.message_handler(commands=['admin_stats'])
def send_admin_report(message):
    if message.chat.id != ADMIN_ID:
        return 
        
    report = stats.get_weekly_report()
    
    text = (
        "📊 **Фрактальный Навигатор: Аналитика**\n\n"
        f"👥 Всего пользователей в БД: `{report['total_users']}`\n"
        f"🟢 Активных сессий: `{report['active_users']}`\n"
        f"🌀 Всего генераций фракталов: `{report['total_generations']}`\n\n"
        "📈 **Популярность режимов:**\n"
    )
    for g_type, count in report['gen_distribution'].items():
        text += f"• `{g_type}`: {count} раз(а)\n"
        
    dates = [item[0] for item in report['user_growth_7d']]
    counts = [item[1] for item in report['user_growth_7d']]
    
    if dates:
        plt.figure(figsize=(6, 4))
        plt.plot(dates, counts, marker='o', color='#2269eb', linewidth=2)
        plt.title("Рост аудитории (последние 7 дней)", fontsize=10)
        plt.xlabel("Дата", fontsize=8)
        plt.ylabel("Новые пользователи", fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=30)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        safe_send_photo(ADMIN_ID, buf, caption=text, parse_mode='Markdown')
        buf.close()
    else:
        safe_api_call(bot.send_message, ADMIN_ID, text + "\n_Данных для построения графика пока недостаточно._", parse_mode='Markdown')

# --- Обработчик ручного запуска рассылки ---
def safe_run_manual_broadcast():
    """Запускает рассылку в отдельном потоке с отловом исключений."""
    try:
        run_broadcast_distribution()
        safe_api_call(bot.send_message, ADMIN_ID, "✅ Ручная рассылка успешно завершена.")
    except Exception as e:
        log("ERROR", "ADMIN", f"Сбой при выполнении ручной рассылки: {e}")
        try:
            safe_api_call(bot.send_message, ADMIN_ID, f"❌ Ошибка при выполнении рассылки: {e}")
        except Exception:
            pass

@bot.message_handler(commands=['admin_broadcast'])
def trigger_manual_broadcast(message):
    if message.chat.id != ADMIN_ID:
        log("WARN", "SECURITY", f"Попытка доступа к ручной рассылке с неавторизованного ID: {message.chat.id}")
        return 
        
    log("INFO", "ADMIN", f"Администратор {ADMIN_ID} запустил ручную рассылку вне расписания.")
    safe_api_call(bot.send_message, ADMIN_ID, "⏳ Инициализирована ручная генерация и рассылка. Это займет некоторое время...")
    
    # Выполнение рассылки в отдельном потоке, чтобы бот продолжал отвечать на сообщения пользователей
    thread = threading.Thread(target=safe_run_manual_broadcast, name="ManualBroadcast")
    thread.daemon = True
    thread.start()

# --- Инициализация и запуск процесса ---
if __name__ == "__main__":
    # Фоновый поток автоматической рассылки по расписанию
    delivery_thread = threading.Thread(target=automated_delivery_loop, name="AutoSend")
    delivery_thread.daemon = True
    delivery_thread.start()
    
    try:
        # Установка статуса "В сети" в описание бота при старте
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/setMyShortDescription",
            json={"short_description": "🟢 В сети. Нажмите /start, чтобы раствориться в бездне."},
            timeout=10
        )
        log("INFO", "SYSTEM", "Статус бота успешно переведен в режим 'Онлайн'.")
    except Exception as e:
        log("ERROR", "SYSTEM", f"Не удалось обновить статус при запуске: {e}")

    # Запуск фонового пинга для сервера-наблюдателя
    ping_thread = threading.Thread(target=heartbeat_loop, name="Heartbeat", daemon=True)
    ping_thread.start()
    
    log("INFO", "SYSTEM", "Бот успешно запущен в космологическом режиме ожидания...")
    
    try:
        bot.infinity_polling()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass
            
        log("INFO", "SYSTEM", "Завершение работы пуллинга. Перевожу статус бота в режим 'Офлайн'...")
        try:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/setMyShortDescription",
                json={"short_description": "🔴 Вне сети. Сервер временно отключен на техническое обслуживание."},
                timeout=5
            )
            log("SUCCESS", "SYSTEM", "Статус офлайна успешно установлен. Процесс завершен.")
        except Exception as e:
            log("ERROR", "SYSTEM", f"Не удалось обновить статус перед выходом: {e}")