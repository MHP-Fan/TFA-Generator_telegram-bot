import os
import io
import time
import random
import secrets
import hashlib
import threading
import signal  
import sys
import numpy as np
import telebot
from telebot import types
from telebot import apihelper
from PIL import Image  # Используем Pillow для пиксель-в-пиксель экспорта и SSAA-фильтрации

# Отключаем GUI для Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Глобальные таймауты для стабильного соединения ---
apihelper.CONNECT_TIMEOUT = 90
apihelper.READ_TIMEOUT = 90

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
            # Аппаратное игнорирование денормализованных чисел (ускоряет глубокий зум)
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
                    self.bot.edit_message_text(text, self.chat_id, self.message_id, parse_mode='Markdown')
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
    """Считывает дизайнерские фразы из файла phrases.txt. При отсутствии берет дефолтные."""
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
        if r_op < 0.40:
            op_list = [OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            child = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "unary", "opcode": op, "child": child}
        else:
            op_list = [OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW]
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
    return has_z and has_c and (num_ops >= 2)

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
            # Аварийный выход по таймауту
            if deadline and time.time() > deadline:
                raise TimeoutError("Превышен жесткий лимит времени вычислений (2 минуты).")
                
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
        # Аварийный выход по таймауту
        if deadline and time.time() > deadline:
            raise TimeoutError("Превышен жесткий лимит времени вычислений (2 минуты).")
            
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

def safe_compute_grid(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, use_double=False, deadline=None):
    if HAS_TORCH and DEVICE.type == 'cuda':
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
    """
    Применяет палитру напрямую к массиву NumPy и возвращает два буфера:
    1. Буфер JPEG (оптимизирован для быстрого inline-просмотра)
    2. Буфер PNG (без сжатия, pixel-perfect для детального зума)
    """
    if cmap is None:
        cmap = CLASSIC_CMAP
        
    body_mask = np.isnan(processed_img)
    clean_img = np.nan_to_num(processed_img, nan=0.0)
    
    rgba_img = cmap(clean_img)
    rgba_img[body_mask] = [0.0, 0.0, 0.0, 1.0] # Тело фрактала красим в черный
    
    rgb_img = (rgba_img[:, :, :3] * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(rgb_img)
    
    # SSAA Даунсамплинг (Lanczos сглаживание)
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
def generate_fractal_pipeline(quality_res=1600, steps=10, progress_callback=None):
    start_time = time.time()
    deadline = start_time + 120.0  # Жесткий лимит 120 секунд (2 минуты)

    rng = random.Random(secrets.randbits(128))
    seed_int = rng.randint(0, 2**128 - 1)
    
    decoder = EntropyDecoder(seed_int)
    ast_tree = generate_ast(decoder, depth=1, max_depth=6) 
    rpn_tokens = []
    ast_to_rpn(ast_tree, rpn_tokens)
    
    while not validate_rpn(rpn_tokens):
        if time.time() > deadline:
            raise TimeoutError("Таймаут безопасности превышен на этапе генерации RPN.")
        seed_int = rng.randint(0, 2**128 - 1)
        decoder = EntropyDecoder(seed_int)
        ast_tree = generate_ast(decoder, depth=1, max_depth=6)
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
        # Для шагов зума всегда используем быстрый float32
        img, x, y = safe_compute_grid(
            xmin, xmax, ymin, ymax, 250, 250, current_max_iter, rpn_tokens, is_julia, c_val, use_double=False, deadline=deadline
        )
        target_x, target_y = find_boundary_point_v2(img, x, y, current_max_iter, rng)
        range_x, range_y = (xmax - xmin)/2.5, (ymax - ymin)/2.5
        xmin, xmax = target_x - range_x/2, target_x + range_x/2
        ymin, ymax = target_y - range_y/2, target_y + range_y/2
        
    final_max_iter = 500
    
    # --- БЫСТРЫЙ ПРЕВЬЮ ПАСС: Ранняя отбраковка (200x200, float32) ---
    if progress_callback:
        progress_callback("⚡ Проверка эстетического потенциала...")
    
    preview_res = 200
    preview_img, _, _ = safe_compute_grid(
        xmin, xmax, ymin, ymax, preview_res, preview_res, final_max_iter, rpn_tokens, is_julia, c_val, use_double=False, deadline=deadline
    )
    preview_processed = apply_adaptive_tonemapping(preview_img, final_max_iter)
    
    if not check_aesthetic_quality(preview_processed):
        log("WARN", "QUALITY", "Фрактал отклонён на стадии быстрого превью.")
        return None, None, None
        
    # --- ФИНАЛЬНЫЙ РЕНДЕР (Адаптивные параметры качества) ---
    # Если запущены на GPU, ставим повышенное качество и SSAA-фильтрацию. На CPU снижаем.
    if HAS_TORCH and DEVICE.type == 'cuda':
        target_res = quality_res # 1600x1600
        ssaa_factor = 1.5       # Рендерим 2400x2400
    else:
        target_res = 1200       # Безопасное разрешение для CPU
        ssaa_factor = 1.0       # Без SSAA во избежание таймаутов
        
    render_res = int(target_res * ssaa_factor)
    
    log("COMPUTE", "GRID", f"Начат рендеринг высокого разрешения {render_res}x{render_res} (Double Precision)...")
    if progress_callback:
        progress_callback(f"🧬 Рендеринг фрактала высокой точности ({target_res}x{target_res})...")
        
    # Финальный рендер требует double precision (use_double=True) для устранения блочной пикселизации
    final_img, _, _ = safe_compute_grid(
        xmin, xmax, ymin, ymax, render_res, render_res, final_max_iter, rpn_tokens, is_julia, c_val, use_double=True, deadline=deadline
    )
    
    processed_img = apply_adaptive_tonemapping(final_img, final_max_iter)
    
    if not check_aesthetic_quality(processed_img):
        log("WARN", "QUALITY", "Фрактал отклонён финальным фильтром эстетического качества.")
        return None, None, None
        
    # Экспорт в два формата
    buf_jpeg, buf_png = export_to_buffers_pil(processed_img, CLASSIC_CMAP, target_res=target_res)
    return buf_jpeg, buf_png, formula_str


# --- Автоматическая рассылка подписчикам ---
def automated_delivery_loop():
    while True:
        # Автоматическая рассылка каждые 2 часа
        time.sleep(7200)
        
        subs = load_subscribers()
        if not subs: continue
            
        log("INFO", "AUTO", f"Начинаю автоматическую отправку фрактала для {len(subs)} подписчиков...")
        try:
            buf_jpeg, buf_png, formula = None, None, None
            for _ in range(15):  
                try:
                    buf_jpeg, buf_png, formula = generate_fractal_pipeline(quality_res=1600, steps=10)
                    if buf_jpeg is not None:
                        break
                except TimeoutError:
                    log("WARN", "AUTO", "Прервано по таймауту в цикле авто-генерации. Ищем дальше...")
                    
            if buf_jpeg is not None:
                for chat_id in list(subs):
                    try:
                        buf_jpeg.seek(0)
                        buf_png.seek(0)
                        
                        bot.send_photo(
                            chat_id,
                            buf_jpeg,
                            caption=(
                                "👁‍🗨 **Внеочередная материализация хаоса**\n\n"
                                "Высший математический порядок пробился сквозь бесконечность.\n"
                                f"Проекция уравнения эволюции:\n`{formula}`\n\n"
                                "⏳ _Вы можете остановить этот поток в меню кнопкой в любой момент._"
                            ),
                            parse_mode='Markdown',
                            timeout=90
                        )
                        
                        buf_png.name = f"fractal_{secrets.token_hex(4)}.png"
                        bot.send_document(
                            chat_id,
                            buf_png,
                            caption="🖼️ **PNG-оригинал без сжатия (для детального зума)**",
                            parse_mode='Markdown',
                            timeout=90
                        )
                        log("SUCCESS", "AUTO", f"Фрактал доставлен подписчику {chat_id}.")
                    except telebot.apihelper.ApiTelegramException as e:
                        if e.error_code in [403, 400]:
                            log("WARN", "AUTO", f"Пользователь {chat_id} заблокировал бота. Удаление подписки.")
                            remove_subscriber(chat_id)
                    except Exception as e:
                        log("ERROR", "AUTO", f"Ошибка отправки пользователю {chat_id}: {e}")
                buf_jpeg.close()
                buf_png.close()
        except Exception as e:
            log("ERROR", "AUTO", f"Критическая ошибка рассылки: {e}")

# --- Динамический интерфейс ---
def get_main_keyboard(chat_id):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn_gen = types.KeyboardButton("🌌 Раствориться в бесконечности")
    
    subs = load_subscribers()
    if chat_id in subs:
        btn_sub = types.KeyboardButton("⏳ Остановить бесконечный поток")
    else:
        btn_sub = types.KeyboardButton("🧿 Запустить бесконечный поток")
        
    btn_batch3 = types.KeyboardButton("🔮 Сгенерировать пакет из 3 фракталов")
    btn_batch5 = types.KeyboardButton("🔮 Сгенерировать пакет из 5 фракталов")
    
    markup.row(btn_gen)
    markup.row(btn_sub)
    markup.row(btn_batch3, btn_batch5)
    return markup

# --- Система отправки пингов на облако (Наблюдатель) ---
def heartbeat_loop():
    while True:
        try:
            requests.get("https://yourusername.pythonanywhere.com/ping", timeout=10)
        except Exception as e:
            log("WARN", "SYSTEM", f"Не удалось отправить пинг на сервер-наблюдатель: {e}")
        time.sleep(60)


# --- Telegram Bot Handlers ---
@bot.message_handler(commands=['start', 'help', 'restart'])
def send_welcome(message):
    try:
        bot.send_message(
            message.chat.id, 
            "«Однажды погрузившись в фрактал, ты больше никогда не остановишься. "
            "Позволь математике растворить тебя в бесконечности иррациональных чисел...»\n\n"
            "👁‍⚙ **Синхронизация интерфейса завершена.** Старые кнопки обновлены.\n"
            "Используйте панель управления ниже для взаимодействия с бесконечностью.", 
            reply_markup=get_main_keyboard(message.chat.id),
            parse_mode='Markdown'
        )
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Ошибка отправки приветствия: {e}")

@bot.message_handler(func=lambda message: message.text == "🌌 Раствориться в бесконечности")
@bot.message_handler(commands=['generate'])
def send_fractal(message):
    chat_id = message.chat.id
    
    status, val = user_manager.try_start_job(chat_id)
    if status == "busy":
        try:
            bot.send_message(chat_id, "⚠️ Вычисления уже запущены. Дождитесь завершения текущего процесса.")
        except Exception:
            pass
        return
    elif status == "cooldown":
        try:
            bot.send_message(chat_id, f"⏳ Пожалуйста, подождите {val:.1f} сек. перед следующей генерацией.")
        except Exception:
            pass
        return

    # КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: try...finally оборачивает абсолютно ВСЁ после try_start_job.
    # Если на этапе send_message упадет прокси, пользователь ГАРАНТИРОВАННО разблокируется.
    try:
        status_msg = bot.send_message(chat_id, "🧬 Инициализация структуры... Настройка математического ядра.")
        log("INFO", "TELEGRAM", f"Пользователь {chat_id} запросил одиночный фрактал.")
        
        updater = ProgressUpdater(bot, chat_id, status_msg.message_id)
        
        buf_jpeg, buf_png, formula = None, None, None
        max_attempts = 15
        
        for attempt in range(1, max_attempts + 1):
            def make_callback(att):
                return lambda text: updater.update(f"🧬 *Попытка {att}/{max_attempts}*\n└ {text}")
            
            updater.update(f"🧬 *Попытка {attempt}/{max_attempts}*\n└ Инициализация матрицы...", force=True)
            
            try:
                buf_jpeg, buf_png, formula = generate_fractal_pipeline(
                    quality_res=1600, 
                    steps=10, 
                    progress_callback=make_callback(attempt)
                )
            except TimeoutError as te:
                log("ERROR", "TIMEOUT", f"Таймаут вычислений на попытке {attempt}: {te}")
                # Если это была последняя попытка, пробрасываем ошибку дальше
                if attempt == max_attempts:
                    raise te
                # Иначе пишем пользователю и пробуем другую координату
                updater.update(f"⚠️ *Попытка {attempt}/{max_attempts}* прервана по таймауту (2 минуты). Ищем другую сингулярность...", force=True)
                time.sleep(1.0)
                continue
            
            if buf_jpeg is not None:
                break
            else:
                updater.update(
                    f"⚠️ *Попытка {attempt}/{max_attempts}* отклонена.\n"
                    f"└ _Точка поля не рекомендована к визуализации (низкая эстетика). Ищем новую сингулярность..._",
                    force=True
                )
                time.sleep(1.0)
        
        if buf_jpeg is None:
            updater.update("👁‍⚙ Математический хаос оказался слишком неустойчив. Повторите попытку прорыва.", force=True)
            return
            
        try:
            bot.delete_message(chat_id, status_msg.message_id)
        except Exception:
            pass
        
        # 1. Отправляем быстрое превью в виде обычной сжатой фотографии
        log("UPLOAD", "TELEGRAM", f"Отправка превью-фото {chat_id}...")
        bot.send_photo(
            chat_id, 
            buf_jpeg, 
            caption=(
                f"🔮 **Погружение совершено.**\n\n"
                f"Хаос упорядочен формулой:\n`{formula}`\n\n"
                f"{get_random_phrase()}"
            ), 
            parse_mode='Markdown',
            reply_markup=get_main_keyboard(chat_id),
            timeout=90
        )
        
        # 2. Следом отправляем оригинальный файл PNG без сжатия в виде Документа
        log("UPLOAD", "TELEGRAM", f"Отправка оригинального PNG-файла {chat_id}...")
        buf_png.name = f"fractal_{secrets.token_hex(4)}.png"
        bot.send_document(
            chat_id,
            buf_png,
            caption="🖼️ **Оригинальная проекция (PNG, без сжатия)**\n└ _Скачайте файл, чтобы рассмотреть микродетали без артефактов зума._",
            parse_mode='Markdown',
            timeout=90
        )
        
        log("SUCCESS", "TELEGRAM", f"Фрактал успешно доставлен пользователю {chat_id} в обоих форматах.")
        buf_jpeg.close()
        buf_png.close()
        
    except TimeoutError:
        log("ERROR", "TELEGRAM", f"Генерация для {chat_id} полностью остановлена из-за превышения лимита времени.")
        try:
            bot.send_message(
                chat_id, 
                "⚠️ **Вычисления прерваны по таймауту (2 минуты).**\n\n"
                "Генерируемое фрактальное уравнение оказалось математически слишком ресурсоемким. "
                "Ваша сессия была завершена во избежание зависания сервера. Пожалуйста, попробуйте снова.",
                reply_markup=get_main_keyboard(chat_id)
            )
        except Exception:
            pass
    except telebot.apihelper.ApiTelegramException as te:
        if te.error_code in [403, 400]:
            log("WARN", "TELEGRAM", f"Пользователь {chat_id} заблокировал бота. Удаление подписки.")
            remove_subscriber(chat_id)
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Сбой отправки: {e}")
        try:
            bot.send_message(chat_id, f"❌ Произошел сбой при генерации фрактала: {str(e)}", reply_markup=get_main_keyboard(chat_id))
        except Exception:
            pass
    finally:
        user_manager.end_job(chat_id)

@bot.message_handler(func=lambda message: message.text in ["🧿 Запустить бесконечный поток", "⏳ Остановить бесконечный поток"])
@bot.message_handler(commands=['subscribe', 'unsubscribe'])
def toggle_subscription(message):
    chat_id = message.chat.id
    try:
        if "Запустить" in message.text or message.text == "/subscribe":
            save_subscriber(chat_id)
            bot.send_message(
                chat_id, 
                "👁‍⚙ **Поток запущен.**\n\nКаждые два часа математическое ядро будет проецировать новую случайную структуру высокой точности прямо в ваше сознание.",
                reply_markup=get_main_keyboard(chat_id)
            )
        else:
            remove_subscriber(chat_id)
            bot.send_message(
                chat_id, 
                "⏳ **Поток приостановлен.**\n\nБесконечность отпускает вас... до следующего ручного погружения.",
                reply_markup=get_main_keyboard(chat_id)
            )
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Ошибка изменения подписки {chat_id}: {e}")

@bot.message_handler(func=lambda message: message.text.startswith("🔮 Сгенерировать пакет"))
def send_batch_fractal(message):
    chat_id = message.chat.id
    num = 5 if "5" in message.text else 3
    
    status, val = user_manager.try_start_job(chat_id)
    if status == "busy":
        try:
            bot.send_message(chat_id, "⚠️ Идет рендеринг предыдущего пакета. Пожалуйста, подождите.")
        except Exception:
            pass
        return
    elif status == "cooldown":
        try:
            bot.send_message(chat_id, f"⏳ Система охлаждается. Попробуйте снова через {val:.1f} сек.")
        except Exception:
            pass
        return

    # Защищенный try...finally блок для пакетного рендера
    try:
        status_msg = bot.send_message(
            chat_id, 
            f"🧬 Инициация каскадного пакета ({num} фрактальных проекций)...\nМатрицы рассчитываются последовательно на CUDA-ядрах."
        )
        log("INFO", "TELEGRAM", f"Пользователь {chat_id} запросил пакет из {num} фракталов.")
        
        updater = ProgressUpdater(bot, chat_id, status_msg.message_id)
        generated = 0
        
        for i in range(num):
            buf_jpeg, buf_png, formula = None, None, None
            max_attempts = 15
            
            for attempt in range(1, max_attempts + 1):
                def make_batch_callback(index, att):
                    return lambda text: updater.update(
                        f"🪐 *Фрактал {index+1} из {num}*\n"
                        f"└ Попытка {att}/{max_attempts}: {text}"
                    )
                
                updater.update(
                    f"🪐 *Фрактал {i+1} из {num}*\n"
                    f"└ Попытка {attempt}/{max_attempts}: Расчёт координат...",
                    force=True
                )
                
                try:
                    buf_jpeg, buf_png, formula = generate_fractal_pipeline(
                        quality_res=1600, 
                        steps=10, 
                        progress_callback=make_batch_callback(i, attempt)
                    )
                except TimeoutError:
                    log("WARN", "TIMEOUT", f"Пакет: Фрактал {i+1} на попытке {attempt} прерван по таймауту.")
                    if attempt == max_attempts:
                        break
                    updater.update(f"⚠️ *Фрактал {i+1} из {num}* (Таймаут). Пробуем другую сингулярность...", force=True)
                    time.sleep(1.0)
                    continue
                
                if buf_jpeg is not None:
                    break
                else:
                    updater.update(
                        f"⚠️ *Фрактал {i+1} из {num}* (Попытка {attempt} отклонена)\n"
                        f"└ _Хаотическая область нестабильна. Пересчет сингулярности..._",
                        force=True
                    )
                    time.sleep(1.0)
            
            if buf_jpeg is not None:
                try:
                    log("UPLOAD", "TELEGRAM", f"Отправка кадра {i+1}/{num} пользователю {chat_id}...")
                    # В пакетах отправляем только JPEG-версии, чтобы не спамить чат файлами
                    bot.send_photo(
                        chat_id,
                        buf_jpeg,
                        caption=f"✨ **Фрактальный слой #{i+1}**\n\nТрансцендентное уравнение эволюции:\n`{formula}`",
                        parse_mode='Markdown',
                        timeout=90
                    )
                    log("SUCCESS", "TELEGRAM", f"Кадр {i+1}/{num} успешно доставлен пользователю {chat_id}.")
                    generated += 1
                except telebot.apihelper.ApiTelegramException as te:
                    if te.error_code in [403, 400]:
                        log("WARN", "TELEGRAM", f"Пользователь {chat_id} заблокировал бота во время пакетного рендеринга. Генерация прервана.")
                        remove_subscriber(chat_id)
                        break  
                except Exception as e:
                    log("ERROR", "TELEGRAM", f"Ошибка отправки кадра {i+1} пакета: {e}")
                finally:
                    buf_jpeg.close()
                    buf_png.close()
                time.sleep(1)
                
        try:
            bot.delete_message(chat_id, status_msg.message_id)
        except Exception:
            pass
            
        if generated == 0:
            bot.send_message(chat_id, "❌ Не удалось пробиться сквозь хаос. Матрица пуста.", reply_markup=get_main_keyboard(chat_id))
        else:
            bot.send_message(chat_id, "🔮 **Каскадный перенос завершен.** Вы растворились во множестве решений.", reply_markup=get_main_keyboard(chat_id))
            
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Критическая ошибка пакетной генерации {chat_id}: {e}")
        try:
            bot.send_message(chat_id, f"❌ Произошел сетевой или вычислительный сбой при генерации пакета: {str(e)}", reply_markup=get_main_keyboard(chat_id))
        except Exception:
            pass
    finally:
        user_manager.end_job(chat_id)

# --- Инициализация и запуск процесса ---
if __name__ == "__main__":
    import requests 
    import signal  
    
    # Фоновый поток автоматической рассылки
    delivery_thread = threading.Thread(target=automated_delivery_loop, name="AutoSend")
    delivery_thread.daemon = True
    delivery_thread.start()
    
    try:
        # При старте сразу ставим статус "В сети" в описание профиля (Bio) бота
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/setMyShortDescription",
            json={"short_description": "🟢 В сети. Нажмите /start, чтобы раствориться в бездне."}
        )
        log("INFO", "SYSTEM", "Статус бота успешно переведен в режим 'Онлайн'.")
    except Exception as e:
        log("ERROR", "SYSTEM", f"Не удалось обновить статус при запуске: {e}")

    # Запуск фонового пинга для сервера-наблюдателя
    ping_thread = threading.Thread(target=heartbeat_loop, name="Heartbeat", daemon=True)
    ping_thread.start()
    
    log("INFO", "SYSTEM", "Бот успешно запущен на локальных ресурсах в космологическом режиме ожидания...")
    
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