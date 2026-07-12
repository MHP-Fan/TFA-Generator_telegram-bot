import os
import sys
import time
import threading
import random
import secrets
import telebot
from telebot import apihelper

# --- Инициализация глобальных сетевых параметров ---
BOT_TOKEN = os.environ.get("BOT_TOKEN", "ВАШ_ТОКЕН_СЮДА")
if BOT_TOKEN == "ВАШ_ТОКЕН_СЮДА" or not BOT_TOKEN:
    raise ValueError("Замените заглушку 'ВАШ_ТОКЕН_СЮДА' на реальный токен от @BotFather!")

ADMIN_ID = int(os.environ.get("ADMIN_ID", "123456789"))

apihelper.CONNECT_TIMEOUT = 90
apihelper.READ_TIMEOUT = 90

bot = telebot.TeleBot(BOT_TOKEN)

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
        
        # Прогрев CUDA на главном потоке для предотвращения дедлоков
        try:
            dummy = torch.zeros(1, device=DEVICE)
            del dummy
            torch.cuda.empty_cache()
            print("[Device] CUDA-контекст успешно прогрет. Защита от дедлока активна.")
        except Exception as e:
            print(f"[Device] Предупреждение при прогреве CUDA: {e}")
    else:
        DEVICE = torch.device('cpu')
        print("[Device] CUDA недоступна. Вычисления переведены на CPU PyTorch.")
except ImportError:
    print("[Device] PyTorch не обнаружен. Вычисления переведены на CPU NumPy.")

# Глобальные блокировки
log_lock = threading.Lock()
render_lock = threading.Lock()

def log(level, section, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    with log_lock:
        print(f"[{timestamp}] [{level:<7}] [{thread_name}] [{section:<8}] {message}", flush=True)

# Сетевой слой с защитой от разрывов
def safe_api_call(func, *args, **kwargs):
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

# Класс для плавного обновления статуса генерации
class ProgressUpdater:
    def __init__(self, bot_instance, chat_id, message_id):
        self.bot = bot_instance
        self.chat_id = chat_id
        self.message_id = message_id
        self.last_text = ""
        self.last_update_time = 0.0
        self.lock = threading.Lock()

    def update(self, text, force=False):
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

# --- Локализация ---
HELP_TEXT_RU = (
    "👁‍⚙ **Фрактальный навигатор — справка**\n\n"
    "Бот генерирует уникальные процедурные фракталы на основе математических формул.\n\n"
    "🎛 **Кнопки управления:**\n"
    "🌌 *Малые расстояния* – стандартное погружение (4-10 шагов).\n"
    "🌀 *Сверхглубокий зум* – глубокое погружение (15-30 шагов).\n"
    "🧿 *Запустить бесконечный поток* – автоматическая рассылка каждые 2 часа.\n"
    "⏳ *Остановить поток* – отключение рассылки.\n"
    "🔮 *Пакет из 3 / 5 фракталов* – последовательная генерация.\n"
    "✍️ *Свой фрактал* – рендеринг по вашей формуле.\n"
    "⚙️ *Настройки* – параметры масштабирования.\n"
    "❓ *Помощь* – справка.\n\n"
    "⚙️ **Технические детали:**\n"
    "• Каждое вычисление ограничено лимитом времени.\n"
    "• Пауза между запросами пользователя — 4 секунды."
)

HELP_TEXT_EN = (
    "👁‍⚙ **Fractal Navigator — Help**\n\n"
    "The bot generates unique procedural fractals using mathematical formulas.\n\n"
    "🎛 **Control Keys:**\n"
    "🌌 *Shallow Zoom* – standard dive (4-10 steps).\n"
    "🌀 *Deep Zoom* – deep dive (15-30 steps).\n"
    "🧿 *Start Infinite Stream* – auto delivery every 2 hours.\n"
    "⏳ *Stop Infinite Stream* – disables delivery.\n"
    "🔮 *Batch of 3 / 5* – sequential generation.\n"
    "✍️ *Custom Fractal* – render by your formula.\n"
    "⚙️ *Settings* – scale parameters.\n"
    "❓ *Help* – help message.\n\n"
    "⚙️ **Technical Details:**\n"
    "• Each calculation is limited by a deadline.\n"
    "• Cooldown between manual requests is 4 seconds."
)

TRANSLATIONS = {
    "ru": {
        "welcome": (
            "«Однажды погрузившись в фрактал, ты больше никогда не остановишься...»\n\n"
            "👁‍⚙ **Синхронизация интерфейса завершена.** Используйте панель ниже."
        ),
        "help": HELP_TEXT_RU,
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
        "rejected_attempt": "⚠️ *Попытка {attempt}/{max_attempts}* отклонена.\n└ _Область не рекомендована к визуализации. Ищем другую..._",
        "timeout_retry": "⚠️ *Попытка {attempt}/{max_attempts}* прервана по таймауту. Пересчет сингулярности...",
        "unstable_chaos": "👁‍⚙ Математический хаос оказался слишком неустойчив. Повторите попытку.",
        "fractal_ready": "🔮 **Погружение совершено.**\n\nХаос упорядочен формулой:\n`{formula}`\n\nКоординаты:\n`xmin = {xmin:.10f}`\n`xmax = {xmax:.10f}`\n`ymin = {ymin:.10f}`\n`ymax = {ymax:.10f}`\n\n{phrase}",
        "png_caption": "🖼️ **Оригинальная проекция (PNG, без сжатия)**\n└ _Скачайте файл, чтобы рассмотреть микродетали._",
        "timeout_error": "⚠️ **Вычисления прерваны по таймауту (2 минуты).**\n\nГенерируемое уравнение оказалось слишком ресурсоемким.",
        "gen_error": "❌ Произошел сбой при генерации фрактала: {error}",
        "sub_started": "👁‍⚙ **Поток запущен.**\n\nКаждые два часа вы будете получать новый случайный фрактал.",
        "sub_stopped": "⏳ **Поток приостановлен.**",
        "batch_init": "🧬 Инициация пакетного рендеринга ({num} проекций).\nРежим масштабирования: **{mode}**.",
        "batch_step": "🪐 *Фрактал {index} из {num}*\n└ Попытка {attempt}/{max_attempts}: {text}",
        "batch_ready": "✨ **Фрактальный слой #{index}**\n\nУравнение:\n`{formula}`\n\nКоординаты:\n`xmin = {xmin:.8f}`\n`xmax = {xmax:.8f}`\n`ymin = {ymin:.8f}`\n`ymax = {ymax:.8f}`",
        "batch_failed": "❌ Не удалось пробиться сквозь хаос.",
        "batch_success": "🔮 **Пакетный перенос завершен.** Все проекции визуализированы.",
        "broadcast_caption": "👁‍🗨 **Плановая материализация хаоса**\n\nПроекция уравнения:\n`{formula}`\n\n`xmin = {xmin:.10f}`\n`xmax = {xmax:.10f}`\n`ymin = {ymin:.10f}`\n`ymax = {ymax:.10f}`",
        "custom_info": "✍️ **Генерация собственного фрактала по формуле**\n\nПришлите ваши параметры ответным сообщением (reply) на это сообщение.",
        "custom_error": "❌ **Ошибка разбора формулы:**\n`{error}`",
        "custom_success": "🎨 **Ваш кастомный фрактал готов!**\n\n`{formula}`"
    },
    "en": {
        "welcome": (
            "“Once you dive into a fractal, you will never stop...”\n\n"
            "👁‍⚙ **Interface synchronization complete.** Use the control panel below."
        ),
        "help": HELP_TEXT_EN,
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
        "settings_desc": "Current batch rendering mode: **{mode}**",
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
        "rejected_attempt": "⚠️ *Attempt {attempt}/{max_attempts}* rejected.",
        "timeout_retry": "⚠️ *Attempt {attempt}/{max_attempts}* timed out.",
        "unstable_chaos": "👁‍⚙ Mathematical chaos proved too unstable. Please try again.",
        "fractal_ready": "🔮 **Dive completed.**\n\n`{formula}`\n\n`xmin = {xmin:.10f}`\n`xmax = {xmax:.10f}`\n`ymin = {ymin:.10f}`\n`ymax = {ymax:.10f}`\n\n{phrase}",
        "png_caption": "🖼️ **Original projection (PNG, uncompressed)**",
        "timeout_error": "⚠️ **Computation timed out (2 minutes).**",
        "gen_error": "❌ Generation failed: {error}",
        "sub_started": "👁‍⚙ **Stream started.**",
        "sub_stopped": "⏳ **Stream paused.**",
        "batch_init": "🧬 Initiating batch rendering ({num} projections).\nZoom mode: **{mode}**.",
        "batch_step": "🪐 *Fractal {index} of {num}*\n└ Attempt {attempt}/{max_attempts}: {text}",
        "batch_ready": "✨ **Fractal Layer #{index}**\n\n`{formula}`",
        "batch_failed": "❌ Failed to break through chaos.",
        "batch_success": "🔮 **Batch transfer complete.**",
        "broadcast_caption": "👁‍🗨 **Scheduled Materialization of Chaos**\n\n`{formula}`",
        "custom_info": "✍️ **Generate Your Own Fractal by Formula**\n\nSend your parameters in a reply to this message.",
        "custom_error": "❌ **Formula parsing error:**\n`{error}`",
        "custom_success": "🎨 **Your custom fractal is ready!**"
    }
}

DEFAULT_PHRASES_EN = [
    "The aesthetics of fractal composition in its pure mathematical form.",
    "The balance of symmetry and asymmetry born of a formula.",
    "Geometry as a way to order visual chaos.",
    "Exploring the plastics and rhythm of complex space."
]

DEFAULT_PHRASES = [
    "Эстетика фрактальной композиции в ее чистом математическом проявлении.",
    "Баланс симметрии и асимметрии, рожденный формулой.",
    "Геометрия как способ упорядочить визуальный хаос.",
    "Исследование пластики и ритма комплексного пространства."
]

PHRASES_FILE = "phrases.txt"

def load_phrases():
    if not os.path.exists(PHRASES_FILE):
        log("WARN", "SYSTEM", f"Файл {PHRASES_FILE} не найден. Используются дефолтные фразы.")
        return DEFAULT_PHRASES
    try:
        with open(PHRASES_FILE, "r", encoding="utf-8") as f:
            phrases = [line.strip() for line in f if line.strip()]
        if phrases:
            log("INFO", "SYSTEM", f"Успешно загружено {len(phrases)} фраз.")
            return phrases
        return DEFAULT_PHRASES
    except Exception as e:
        log("ERROR", "SYSTEM", f"Ошибка при чтении {PHRASES_FILE}: {e}")
        return DEFAULT_PHRASES

bot_phrases = load_phrases()

def get_random_phrase():
    return random.choice(bot_phrases)