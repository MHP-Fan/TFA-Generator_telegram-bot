import os
import json
import sqlite3
import threading
import time
from datetime import datetime
from config import log

# --- База данных статистики ---
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS broadcast_deliveries (
                    epoch INTEGER,
                    chat_id INTEGER,
                    timestamp TEXT,
                    PRIMARY KEY (epoch, chat_id)
                )
            """)
            conn.commit()
            conn.close()

    def is_broadcast_delivered(self, epoch, chat_id):
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM broadcast_deliveries WHERE epoch = ? AND chat_id = ?",
                (int(epoch), int(chat_id))
            )
            res = cursor.fetchone()
            conn.close()
            return res is not None

    def record_broadcast_delivery(self, epoch, chat_id):
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT OR IGNORE INTO broadcast_deliveries (epoch, chat_id, timestamp) VALUES (?, ?, ?)",
                (int(epoch), int(chat_id), now_str)
            )
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

# --- Менеджер подписок ---
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

# --- Настройки пользователей ---
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

def get_user_lang(message_or_chat_id):
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

# --- Состояние запущенных задач (Кулдауны) ---
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
                return "cooldown", cooldown_sec - (now - last_time)
            
            if chat_id in self.active_jobs:
                return "busy", None
            
            self.active_jobs.add(chat_id)
            self.last_request[chat_id] = now
            return "ok", None
            
    def end_job(self, chat_id):
        with self.lock:
            if chat_id in self.active_jobs:
                self.active_jobs.remove(chat_id)

user_manager = UserManager()