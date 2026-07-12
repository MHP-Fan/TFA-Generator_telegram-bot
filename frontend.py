import os
import time
import secrets
import hashlib
import random
import numpy as np

# Принудительный фоновый режим рендеринга Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Определение аппаратной конфигурации ---
HAS_TORCH = False
DEVICE = None

try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        torch.cuda.empty_cache()
        print(f"[Device] Обнаружена видеокарта CUDA: {torch.cuda.get_device_name(0)}")
        print("[Device] CUDA-вычислитель успешно инициализирован.\n")
    else:
        DEVICE = torch.device('cpu')
        print("[Device] Видеокарта NVIDIA не найдена. Расчеты будут выполняться на CPU PyTorch.\n")
except ImportError:
    print("[Device] Библиотека PyTorch отсутствует. Переключение на вычисления через NumPy.\n")


# --- Константы виртуальной машины RPN ---
VAR_Z, VAR_C, CONST = 0, 1, 2
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW = 3, 4, 5, 6, 7
OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM = 8, 9, 10, 11, 12, 13, 14, 15

EPS_REG = 1e-20  # Барьер для исключения деления на ноль


def fast_uniform_filter(arr, size=15):
    """
    Вычисляет среднее значение в скользящем окне на базе интегрального представления.
    Использует защищенное нулевое дополнение для исключения выхода индексов за границы.
    """
    sz = size // 2
    # Граничное дополнение исходного массива
    padded = np.pad(arr, sz, mode='edge')
    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    # Дополнение массива префиксных сумм нулями слева и сверху для корректной разности
    cumsum_padded = np.pad(cumsum, ((1, 0), (1, 0)), mode='constant', constant_values=0)
    
    h, w = arr.shape
    # Сумма элементов внутри скользящего окна размером size x size
    total = (cumsum_padded[size:h + size, size:w + size]
             - cumsum_padded[0:h, size:w + size]
             - cumsum_padded[size:h + size, 0:w]
             + cumsum_padded[0:h, 0:w])
             
    return total / (size * size)


# --- Настройки цветовой палитры ---
def get_classic_colormap():
    colors = [
        (0.0, '#000105'),    # Глубокий черный космос
        (0.12, '#01061c'),   # Иссиня-черный
        (0.32, '#041d5e'),   # Королевский синий
        (0.55, '#2269eb'),   # Электрический синий
        (0.72, '#82b5ff'),   # Неоновый голубой
        (0.85, '#ffffff'),   # Белый разряд
        (0.92, '#ffaa00'),   # Яркое золото
        (0.97, '#ff3700'),   # Оранжево-красный
        (1.0, '#000000')     # Переход в черное тело
    ]
    return LinearSegmentedColormap.from_list("DynamicProceduralMap", colors, N=2048)


# --- Класс декодера энтропии ---
class EntropyDecoder:
    """
    Обеспечивает воспроизводимое преобразование 512-битного сида
    в псевдослучайный поток данных через последовательное хэширование SHA-512.
    """
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
        """Генерация вещественного числа в диапазоне [0.0, 1.0)"""
        b0 = self.get_next_byte()
        b1 = self.get_next_byte()
        b2 = self.get_next_byte()
        b3 = self.get_next_byte()
        val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        return val / 4294967296.0


# --- Построение процедурного дерева выражений (AST -> RPN) ---
def generate_ast(decoder, depth, max_depth):
    """
    Рекурсивно формирует синтаксическое дерево. 
    Вероятность образования терминальных ветвей повышается по мере углубления.
    """
    p_terminal = min(1.0, (depth - 1) / (max_depth - 1)) if depth > 1 else 0.0
    
    if decoder.get_float() < p_terminal:
        r = decoder.get_float()
        if r < 0.45:
            return {"type": "terminal", "opcode": VAR_Z, "val": None}
        elif r < 0.85:
            return {"type": "terminal", "opcode": VAR_C, "val": None}
        else:
            real = -1.5 + 3.0 * decoder.get_float()
            imag = -1.5 + 3.0 * decoder.get_float()
            return {"type": "terminal", "opcode": CONST, "val": complex(real, imag)}
    else:
        r_op = decoder.get_float()
        if r_op < 0.40:  # Унарный оператор
            op_list = [OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            child = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "unary", "opcode": op, "child": child}
        else:            # Бинарный оператор
            op_list = [OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            left = generate_ast(decoder, depth + 1, max_depth)
            right = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "binary", "opcode": op, "left": left, "right": right}


def ast_to_rpn(node, rpn):
    """Компилирует AST дерево в линейный стек обратной польской записи"""
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
    """Статическая семантическая верификация формулы на непротиворечивость"""
    has_z = any(t == 0 and op == VAR_Z for t, op, _ in rpn)
    has_c = any(t == 0 and op == VAR_C for t, op, _ in rpn)
    num_ops = sum(1 for t, _, _ in rpn if t in (1, 2))
    return has_z and has_c and (num_ops >= 2)


def rpn_to_str(rpn):
    """Транслирует стек RPN в читаемую строку математического уравнения"""
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


# --- Вычислительные интерпретаторы (PyTorch & NumPy) ---
def evaluate_rpn_pytorch(rpn, Z, C, device):
    stack = []
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z:
                stack.append(Z)
            elif op == VAR_C:
                stack.append(C)
            elif op == CONST:
                stack.append(torch.tensor(val, dtype=torch.complex128, device=device))
        elif t_type == 1:
            A = stack.pop()
            if op == OP_SIN:
                A_stab = torch.complex(A.real, torch.clamp(A.imag, -15.0, 15.0))
                stack.append(torch.sin(A_stab))
            elif op == OP_COS:
                A_stab = torch.complex(A.real, torch.clamp(A.imag, -15.0, 15.0))
                stack.append(torch.cos(A_stab))
            elif op == OP_EXP:
                A_stab = torch.complex(torch.clamp(A.real, -15.0, 15.0), A.imag)
                stack.append(torch.exp(A_stab))
            elif op == OP_LN:
                mag_sq = A.real**2 + A.imag**2 + EPS_REG
                real = 0.5 * torch.log(mag_sq)
                imag = torch.atan2(A.imag, A.real)
                stack.append(torch.complex(real, imag))
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
            if op == OP_ADD:
                stack.append(A + B)
            elif op == OP_SUB:
                stack.append(A - B)
            elif op == OP_MUL:
                stack.append(A * B)
            elif op == OP_DIV:
                denom = B.real**2 + B.imag**2 + EPS_REG
                real = (A.real * B.real + A.imag * B.imag) / denom
                imag = (A.imag * B.real - A.real * B.imag) / denom
                stack.append(torch.complex(real, imag))
            elif op == OP_POW:
                mag_sq = A.real**2 + A.imag**2 + EPS_REG
                ln_real = 0.5 * torch.log(mag_sq)
                ln_imag = torch.atan2(A.imag, A.real)
                ln_A = torch.complex(ln_real, ln_imag)
                prod = B * ln_A
                prod_stab = torch.complex(torch.clamp(prod.real, -15.0, 15.0), prod.imag)
                stack.append(torch.exp(prod_stab))
                
    Z_next = stack[0]
    anomalies = torch.isnan(Z_next) | torch.isinf(Z_next)
    if torch.any(anomalies):
        Z_next = torch.where(anomalies, torch.tensor(1e5 + 0j, dtype=torch.complex128, device=device), Z_next)
    return Z_next


def evaluate_rpn_numpy(rpn, Z, C):
    stack = []
    for t_type, op, val in rpn:
        if t_type == 0:
            if op == VAR_Z: stack.append(Z)
            elif op == VAR_C: stack.append(C)
            elif op == CONST: stack.append(val)
        elif t_type == 1:
            A = stack.pop()
            with np.errstate(invalid='ignore', over='ignore'):
                if op == OP_SIN:
                    A_stab = np.real(A) + 1j * np.clip(np.imag(A), -15.0, 15.0)
                    stack.append(np.sin(A_stab))
                elif op == OP_COS:
                    A_stab = np.real(A) + 1j * np.clip(np.imag(A), -15.0, 15.0)
                    stack.append(np.cos(A_stab))
                elif op == OP_EXP:
                    A_stab = np.clip(np.real(A), -15.0, 15.0) + 1j * np.imag(A)
                    stack.append(np.exp(A_stab))
                elif op == OP_LN:
                    mag_sq = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    real = 0.5 * np.log(mag_sq)
                    imag = np.arctan2(np.imag(A), np.real(A))
                    stack.append(real + 1j * imag)
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
                if op == OP_ADD:
                    stack.append(A + B)
                elif op == OP_SUB:
                    stack.append(A - B)
                elif op == OP_MUL:
                    stack.append(A * B)
                elif op == OP_DIV:
                    denom = np.real(B)**2 + np.imag(B)**2 + EPS_REG
                    real = (np.real(A)*np.real(B) + np.imag(A)*np.imag(B)) / denom
                    imag = (np.imag(A)*np.real(B) - np.real(A)*np.imag(B)) / denom
                    stack.append(real + 1j * imag)
                elif op == OP_POW:
                    mag_sq = np.real(A)**2 + np.imag(A)**2 + EPS_REG
                    ln_real = 0.5 * np.log(mag_sq)
                    ln_imag = np.arctan2(np.imag(A), np.real(A))
                    ln_A = ln_real + 1j * ln_imag
                    prod = B * ln_A
                    prod_stab = np.clip(np.real(prod), -15.0, 15.0) + 1j * np.imag(prod)
                    stack.append(np.exp(prod_stab))
                    
    Z_next = stack[0]
    anomalies = np.isnan(Z_next) | np.isinf(Z_next)
    if np.any(anomalies):
        Z_next = np.where(anomalies, 1e5 + 0j, Z_next)
    return Z_next


# --- Универсальный итератор сетки (Unified Escape-Attractor Metric) ---
def compute_procedural_grid_pytorch(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, device):
    x = torch.linspace(xmin, xmax, width, dtype=torch.float64, device=device)
    y = torch.linspace(ymin, ymax, height, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    C = torch.complex(X, Y)
    
    if is_julia:
        Z = C.clone()
        C_param = torch.tensor(c, dtype=torch.complex128, device=device)
    else:
        Z = torch.zeros_like(C)
        C_param = C
        
    img = torch.zeros(C.shape, dtype=torch.float64, device=device)
    mask = torch.ones(C.shape, dtype=torch.bool, device=device)
    
    Z_prev = torch.zeros_like(C)
    Z_prev2 = torch.zeros_like(C)
    
    R_esc_sq = 1e8
    eps_att_sq = 1e-12
    
    with torch.no_grad():
        for i in range(max_iter):
            Z_next = evaluate_rpn_pytorch(rpn, Z, C_param, device)
            
            mag_sq = Z_next.real**2 + Z_next.imag**2
            escaped = mag_sq > R_esc_sq
            
            dist_prev_sq = (Z_next.real - Z_prev.real)**2 + (Z_next.imag - Z_prev.imag)**2
            dist_prev2_sq = (Z_next.real - Z_prev2.real)**2 + (Z_next.imag - Z_prev2.imag)**2
            attracted = (dist_prev_sq < eps_att_sq) | (dist_prev2_sq < eps_att_sq)
            
            finished = escaped | attracted
            newly_finished = finished & mask
            
            if torch.any(newly_finished):
                z_mag = torch.sqrt(mag_sq[newly_finished])
                z_mag = torch.clamp(z_mag, min=1.001)
                z_prev_mag = torch.clamp(torch.sqrt(Z.real**2 + Z.imag**2)[newly_finished], min=1.001)
                alpha = torch.clamp(torch.log(z_mag) / (torch.log(z_prev_mag) + 1e-20), min=1.1)
                nu = torch.log(torch.log(z_mag)) / torch.log(alpha)
                
                esc_subset = escaped[newly_finished]
                val = torch.where(esc_subset, i + 1.0 - nu, torch.tensor(float(i), dtype=torch.float64, device=device))
                img[newly_finished] = val
                
            mask = mask & ~finished
            if not torch.any(mask):
                break
                
            Z_prev2 = Z_prev.clone()
            Z_prev = Z.clone()
            Z = Z_next
            
        img[mask] = max_iter
    return img.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy()


def compute_procedural_grid_numpy(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    if is_julia:
        Z = C.copy()
        C_param = c
    else:
        Z = np.zeros_like(C)
        C_param = C
        
    img = np.zeros(C.shape, dtype=float)
    mask = np.ones(C.shape, dtype=bool)
    
    Z_prev = np.zeros_like(C)
    Z_prev2 = np.zeros_like(C)
    
    R_esc_sq = 1e8
    eps_att_sq = 1e-12
    
    for i in range(max_iter):
        Z_next = evaluate_rpn_numpy(rpn, Z, C_param)
        
        mag_sq = np.real(Z_next)**2 + np.imag(Z_next)**2
        escaped = mag_sq > R_esc_sq
        
        dist_prev_sq = np.real(Z_next - Z_prev)**2 + np.imag(Z_next - Z_prev)**2
        dist_prev2_sq = np.real(Z_next - Z_prev2)**2 + np.imag(Z_next - Z_prev2)**2
        attracted = (dist_prev_sq < eps_att_sq) | (dist_prev2_sq < eps_att_sq)
        
        finished = escaped | attracted
        newly_finished = finished & mask
        
        if np.any(newly_finished):
            z_mag = np.sqrt(mag_sq[newly_finished])
            z_mag = np.maximum(z_mag, 1.001)
            prev_mag = np.maximum(np.sqrt(np.real(Z)**2 + np.imag(Z)**2)[newly_finished], 1.001)
            alpha = np.maximum(np.log(z_mag) / (np.log(prev_mag) + 1e-20), 1.1)
            nu = np.log(np.log(z_mag)) / np.log(alpha)
            
            esc_subset = escaped[newly_finished]
            val = np.where(esc_subset, i + 1.0 - nu, float(i))
            img[newly_finished] = val
            
        mask = mask & ~finished
        if not np.any(mask):
            break
            
        Z_prev2 = Z_prev.copy()
        Z_prev = Z.copy()
        Z = Z_next
        
    img[mask] = max_iter
    return img, x, y


class DeviceManager:
    def __init__(self):
        self.has_torch = HAS_TORCH
        self.device = DEVICE


def safe_compute_grid(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, device_manager):
    if device_manager.has_torch and device_manager.device.type == 'cuda':
        try:
            return compute_procedural_grid_pytorch(
                xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c, device_manager.device
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print("[Device Error] CUDA Out of Memory. Переключение вычислительного ядра на CPU.")
                width = min(width, 350)
                height = min(height, 350)
            else:
                print(f"[Device Error] Системная ошибка PyTorch: {e}. Принудительный откат на NumPy.")
                
    return compute_procedural_grid_numpy(xmin, xmax, ymin, ymax, width, height, max_iter, rpn, is_julia, c)


# --- Модуль позиционирования Boundary Search 2.0 (Aesthetic Attraction Index) ---
def find_boundary_point_v2(img, x, y, max_iter, rng):
    dy, dx = np.gradient(img)
    grad = np.sqrt(dx**2 + dy**2)
    grad_norm = grad / (np.max(grad) + 1e-8)
    
    # Индекс фазовой разметки (1 - граница перехода, 0 - пустое пространство)
    body_mask = (img >= max_iter - 1.0).astype(float)
    m_bar = fast_uniform_filter(body_mask, size=15)
    phi = 1.0 - 2.0 * np.abs(m_bar - 0.5)
    
    score_map = grad_norm * phi
    
    threshold = np.percentile(score_map, 90)
    candidate_mask = score_map >= threshold
    indices = np.argwhere(candidate_mask)
    
    if len(indices) == 0:
        indices = np.argwhere(score_map > 0)
    if len(indices) == 0:
        return x[len(x)//2], y[len(y)//2]
        
    idx = indices[rng.randint(0, len(indices) - 1)]
    return x[idx[1]], y[idx[0]]


def find_highly_decorated_c_v2(rpn, rng, device_manager):
    """
    Сканирует параметрическое пространство уравнения в режиме Мандельброта,
    чтобы локализовать нетривиальную константу c для множества Жюлиа.
    """
    xmin, xmax = -2.0, 2.0
    ymin, ymax = -2.0, 2.0
    zoom_factor = 2.5
    
    for step in range(4):
        img, x, y = safe_compute_grid(
            xmin, xmax, ymin, ymax, 
            width=150, height=150, max_iter=100,
            rpn=rpn, is_julia=False, c=0j, device_manager=device_manager
        )
        target_x, target_y = find_boundary_point_v2(img, x, y, 100, rng)
        
        range_x = (xmax - xmin) / zoom_factor
        range_y = (ymax - ymin) / zoom_factor
        xmin = target_x - range_x / 2.0
        xmax = target_x + range_x / 2.0
        ymin = target_y - range_y / 2.0
        ymax = target_y + range_y / 2.0
        
    return complex(target_x, target_y)


# --- Адаптивный тонокомпрессор яркости (Luminance Auto-Gain) ---
def apply_adaptive_tonemapping(img, max_iter):
    img_corrected = np.copy(img)
    body_mask = img >= max_iter - 1.0
    non_body_mask = ~body_mask
    
    if not np.any(non_body_mask):
        return None  # Дегенеративный пустой кадр
        
    non_body_vals = img_corrected[non_body_mask]
    v_min = np.min(non_body_vals)
    v_max = np.max(non_body_vals)
    
    if v_max > v_min:
        v_norm = (non_body_vals - v_min) / (v_max - v_min)
    else:
        v_norm = np.ones_like(non_body_vals)
        
    median_val = np.median(v_norm)
    gamma_base = 0.45
    
    # Динамическая экспозиция: сжатие диапазона при смещении медианы к темноте
    gamma_dyn = gamma_base * ((median_val + 1e-5) / 0.5) ** 0.65
    gamma_dyn = np.clip(gamma_dyn, 0.15, 0.45)
    
    w = 0.7
    a = 15.0
    pow_transform = np.power(v_norm, gamma_dyn)
    log_transform = np.log1p(a * v_norm) / np.log1p(a)
    
    v_final = w * pow_transform + (1.0 - w) * log_transform
    
    img_corrected[non_body_mask] = v_final
    img_corrected[body_mask] = np.nan  # Исключаем тело фрактала из раскраски (чистый черный цвет)
    
    return img_corrected


# --- Автоматический фильтр эстетики и контроля качества ---
def check_aesthetic_quality(processed_img):
    if processed_img is None:
        return False
        
    body_mask = np.isnan(processed_img)
    body_ratio = np.sum(body_mask) / body_mask.size
    
    # Кадр не должен быть сплошным черным пятном или пустым градиентом
    if not (0.015 < body_ratio < 0.65):
        return False
        
    non_body = processed_img[~body_mask]
    if non_body.size == 0:
        return False
        
    std_val = np.std(non_body)
    if std_val < 0.12:  # Исключаем слишком блеклые, серые и неконтрастные структуры
        return False
        
    return True


# --- Главный управляющий конвейер (Orchestrator Main Loop) ---
def main():
    output_dir = "Photos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[System] Создана папка сохранения: {os.path.abspath(output_dir)}")
        
    device_manager = DeviceManager()
    cmap_obj = get_classic_colormap()
    cmap_obj.set_bad(color='black')
    
    zoom_factor = 2.5
    print("\n[Start] Запуск процедурного фрактального синтезатора.")
    print("        Программа работает в непрерывном цикле. Остановка: Ctrl+C.\n")
    
    saved_counter = 1
    attempt_counter = 1
    current_seed_int = secrets.randbits(512)
    
    try:
        while True:
            # Генерация лавинообразного смещения сида через SHA-512
            hash_bytes = hashlib.sha512(str(current_seed_int).encode()).digest()
            seed_int = int.from_bytes(hash_bytes, byteorder='big')
            seed_hex = f"{seed_int:0128x}"
            rng = random.Random(seed_int)
            
            # Настройка параметров прохода
            n_steps = rng.randint(5, 12)  # Глубина зум-траектории
            is_julia = (rng.randint(0, 1) == 1) or (seed_int % 2 == 1)
            
            # Декодирование процедурного AST и компиляция в RPN
            decoder = EntropyDecoder(seed_int)
            ast_tree = generate_ast(decoder, depth=1, max_depth=5)
            rpn_tokens = []
            ast_to_rpn(ast_tree, rpn_tokens)
            
            # Проверка семантической нетривиальности уравнения
            if not validate_rpn(rpn_tokens):
                current_seed_int = seed_int
                continue
                
            formula_str = rpn_to_str(rpn_tokens)
            
            # Поиск константы для множества Жюлиа
            c_val = 0j
            if is_julia:
                c_val = find_highly_decorated_c_v2(rpn_tokens, rng, device_manager)
                
            # Начальные пространственные координаты
            xmin, xmax = -2.0, 2.0
            ymin, ymax = -2.0, 2.0
            
            # Итерационное приближение по градиентной траектории границы перехода
            for step in range(1, n_steps + 1):
                current_max_iter = 120 + step * 70
                
                img, x, y = safe_compute_grid(
                    xmin, xmax, ymin, ymax, 
                    width=220, height=220, max_iter=current_max_iter,
                    rpn=rpn_tokens, is_julia=is_julia, c=c_val, device_manager=device_manager
                )
                
                target_x, target_y = find_boundary_point_v2(img, x, y, current_max_iter, rng)
                
                range_x = (xmax - xmin) / zoom_factor
                range_y = (ymax - ymin) / zoom_factor
                xmin = target_x - range_x / 2.0
                xmax = target_x + range_x / 2.0
                ymin = target_y - range_y / 2.0
                ymax = target_y + range_y / 2.0
                
            # Финальный высокоточный рендеринг картины
            final_max_iter = 350 + n_steps * 250
            final_img, _, _ = safe_compute_grid(
                xmin, xmax, ymin, ymax, 
                width=1200, height=1200, max_iter=final_max_iter,
                rpn=rpn_tokens, is_julia=is_julia, c=c_val, device_manager=device_manager
            )
            
            # Адаптивное сжатие яркости кадра (Auto-Gain)
            processed_img = apply_adaptive_tonemapping(final_img, final_max_iter)
            
            # Прохождение эстетического фильтра контроля качества
            if not check_aesthetic_quality(processed_img):
                attempt_counter += 1
                current_seed_int = seed_int
                continue
                
            # Рендеринг в Matplotlib и сохранение файла
            fig = plt.figure(figsize=(12, 12), facecolor='black')
            plt.imshow(processed_img, cmap=cmap_obj, extent=[xmin, xmax, ymin, ymax], origin='lower')
            plt.axis('off')
            plt.tight_layout()
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            milliseconds = int((time.time() % 1) * 1000)
            short_seed = seed_hex[:12]
            total_zoom = (4.0 / (xmax - xmin))
            
            filename = f"fractal_{timestamp}_{milliseconds:03d}_{short_seed}_zoom{n_steps}.png"
            filepath = os.path.join(output_dir, filename)
            
            plt.savefig(filepath, facecolor='black', edgecolor='none', bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)
            
            if device_manager.has_torch and device_manager.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            space_mode = "Julia (Dynamical)" if is_julia else "Mandelbrot (Parameter)"
            print(f"[{saved_counter}] Сохранен: {filename}")
            print(f"    Зум: {total_zoom:,.0f}x | Пространство: {space_mode}")
            print(f"    Синтезированное уравнение: Z_next = {formula_str}")
            if is_julia:
                print(f"    Комплексный параметр: c = {c_val.real:.5f} + {c_val.imag:.5f}j")
            print(f"    Попыток оптимизации кадра: {attempt_counter}\n")
            
            saved_counter += 1
            attempt_counter = 1
            current_seed_int = seed_int
            
    except KeyboardInterrupt:
        print("\n[Stop] Процесс прерван пользователем. Синтез завершен.")


if __name__ == "__main__":
    main()