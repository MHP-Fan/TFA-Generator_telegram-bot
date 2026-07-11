import os
import io
import time
import random
import secrets
import hashlib
import numpy as np
from PIL import Image

# =====================================================================
#                         НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ
# =====================================================================
RESOLUTION = 1600         # Разрешение выходного изображения (квадрат RESOLUTION x RESOLUTION)
MAX_ITER = 60             # Максимальное количество итераций фрактала
START_TRAP_ITER = 1       # С какой итерации ловить орбиту. 
                          # 0 - центр останется исходным фото.
                          # 1 и более - полная абстрактная фрактализация.
TRAP_SIZE = 1.3           # Размер ловушки на комплексной плоскости (оптимально от 1.0 до 2.0)
VIEW_ZOOM = 1.8           # Масштаб просмотра (границы от -VIEW_ZOOM до VIEW_ZOOM)
# =====================================================================

# Константы RPN-вычислителя
VAR_Z, VAR_C, CONST = 0, 1, 2
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW = 3, 4, 5, 6, 7
OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM = 8, 9, 10, 11, 12, 13, 14, 15
EPS_REG = 1e-20

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
        if r_op < 0.35:
            op_list = [OP_SIN, OP_COS, OP_EXP, OP_LN, OP_ABS, OP_CONJ, OP_INV, OP_SIGM]
            op = op_list[decoder.get_next_byte() % len(op_list)]
            child = generate_ast(decoder, depth + 1, max_depth)
            return {"type": "unary", "opcode": op, "child": child}
        else:
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

def evaluate_rpn_numpy(rpn, Z, C):
    stack = []
    complex_dtype = np.complex128
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

def find_input_image():
    valid_extensions = ('.png', '.jpg', '.jpeg')
    for f in os.listdir('.'):
        if f.lower().startswith('input') and f.lower().endswith(valid_extensions):
            return f
    return None

def compute_orbit_trap(photo_path, rpn, res, max_iter):
    # Загружаем изображение-источник
    img_photo = Image.open(photo_path).convert("RGB")
    photo_arr = np.array(img_photo).astype(np.float32) / 255.0
    p_h, p_w, _ = photo_arr.shape

    # Координатная сетка
    xmin, xmax = -VIEW_ZOOM, VIEW_ZOOM
    ymin, ymax = -VIEW_ZOOM, VIEW_ZOOM
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    X, Y = np.meshgrid(x, y)
    
    # Инициализация фрактального пространства (Mandelbrot style: Z0 = 0, C = grid)
    C = (X + 1j * Y).astype(np.complex128)
    Z = np.zeros_like(C)
    
    # Массив для хранения результирующих RGB пикселей
    output_image = np.zeros((res, res, 3), dtype=np.float32)
    not_trapped = np.ones(C.shape, dtype=bool)
    
    # Размер и границы ловушки на комплексной плоскости
    trap_limit = TRAP_SIZE
    trap_xmin, trap_xmax = -trap_limit, trap_limit
    trap_ymin, trap_ymax = -trap_limit, trap_limit

    print("📊 Рассчитываем итерации фрактала...")
    for i in range(max_iter):
        Z = evaluate_rpn_numpy(rpn, Z, C)
        
        # Начинаем захват орбиты только со START_TRAP_ITER
        if i < START_TRAP_ITER:
            continue
            
        real_z = np.real(Z)
        imag_z = np.imag(Z)
        
        # Проверяем попадание точек в ловушку
        in_trap = (
            (real_z >= trap_xmin) & (real_z <= trap_xmax) &
            (imag_z >= trap_ymin) & (imag_z <= trap_ymax)
        )
        
        hit_mask = in_trap & not_trapped
        
        if np.any(hit_mask):
            # Переводим комплексные координаты попавших точек в нормализованные текстурные (U, V)
            u = (real_z[hit_mask] - trap_xmin) / (trap_xmax - trap_xmin)
            # v разворачиваем (1.0 - v), чтобы избежать вертикального переворота фотографии
            v = 1.0 - (imag_z[hit_mask] - trap_ymin) / (trap_ymax - trap_ymin)
            
            # Находим координаты соответствующих пикселей на оригинальном фото
            px_x = np.clip((u * (p_w - 1)).astype(int), 0, p_w - 1)
            px_y = np.clip((v * (p_h - 1)).astype(int), 0, p_h - 1)
            
            # Получаем цвета пикселей
            colors = photo_arr[px_y, px_x]
            
            # Эстетический эффект: легкое затухание цвета на глубоких итерациях
            depth_factor = 0.98 ** (i - START_TRAP_ITER)
            output_image[hit_mask] = colors * depth_factor
            
            # Исключаем пиксель из дальнейшей обработки
            not_trapped[hit_mask] = False
            
        if not np.any(not_trapped):
            break

    # Оформляем фон для областей, которые не попали в ловушку
    # Создаем мягкий космический градиент на основе финального состояния Z
    if np.any(not_trapped):
        escape_val = np.nan_to_num(np.abs(Z[not_trapped]))
        norm_val = np.clip(escape_val / 10.0, 0.0, 1.0)
        
        # Генерируем глубокие оттенки синего и фиолетового
        output_image[not_trapped, 0] = norm_val * 0.05  # R
        output_image[not_trapped, 1] = norm_val * 0.08  # G
        output_image[not_trapped, 2] = norm_val * 0.15  # B

    return (output_image * 255.0).astype(np.uint8)

def main():
    print("👁‍⚙️ Инициализация локального фрактального процессора...")
    
    input_img = find_input_image()
    if not input_img:
        print("❌ Ошибка: В папке скрипта не найден файл input.png или input.jpg!")
        print("Положите картинку в корневую директорию рядом со скриптом.")
        return

    print(f"📥 Обнаружен входной файл: {input_img}")
    
    # Генерация случайного 512-битного сида и формулы
    rng = random.Random(secrets.randbits(128))
    seed_int = rng.randint(0, 2**128 - 1)
    decoder = EntropyDecoder(seed_int)
    
    print("🔮 Синтезируем уникальное математическое уравнение...")
    rpn_tokens = []
    while not validate_rpn(rpn_tokens):
        ast_tree = generate_ast(decoder, depth=1, max_depth=4)
        rpn_tokens = []
        ast_to_rpn(ast_tree, rpn_tokens)
        
    formula_str = rpn_to_str(rpn_tokens)
    print(f"🧬 Сгенерированная формула эволюции:\n👉 {formula_str}\n")
    
    start_time = time.time()
    
    # Рендеринг
    fractal_rgb = compute_orbit_trap(input_img, rpn_tokens, RESOLUTION, MAX_ITER)
    
    # Сохранение результата
    output_filename = "output_fractal.png"
    out_img = Image.fromarray(fractal_rgb)
    out_img.save(output_filename, format="PNG")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Готово! Фрактал успешно материализован.")
    print(f"⏱ Время расчета: {elapsed:.2f} сек.")
    print(f"💾 Результат сохранен в: {output_filename}")

if __name__ == "__main__":
    main()