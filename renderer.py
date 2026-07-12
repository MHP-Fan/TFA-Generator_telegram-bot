import time
import random
import gc
import io
import secrets
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from config import HAS_TORCH, DEVICE, render_lock, log
from math_engine import (
    EntropyDecoder, generate_ast, ast_to_rpn, validate_rpn, rpn_to_str,
    evaluate_rpn_pytorch, evaluate_rpn_numpy, EPS_REG
)

if HAS_TORCH:
    import torch

# Фильтрация и цветовые карты
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
    # Мы скорректировали опорные цвета на низких значениях (0.0 - 0.32),
    # заменив почти черные пиксели на глубокий, но хорошо видимый космический синий.
    colors = [
        (0.0, '#070f2b'),   # Глубокий полночно-синий (вместо #000105). Отчетливо виден на фоне тела.
        (0.12, '#111f4d'),  # Насыщенный темно-синий (вместо #01061c).
        (0.32, '#1b3273'),  # Выразительный синий средней глубины (вместо #041d5e).
        (0.55, '#3c79e6'),  # Небесно-голубой
        (0.72, '#82b5ff'),  # Светло-голубой
        (0.85, '#ffffff'),  # Белые блики
        (0.92, '#ffaa00'),  # Теплый золотой
        (0.97, '#ff3700'),  # Огненный оранжевый
        (1.0, '#000000')    # Черный контур
    ]
    return LinearSegmentedColormap.from_list("DynamicMap", colors, N=2048)

CLASSIC_CMAP = get_classic_colormap()

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
                raise TimeoutError("Превышен лимит времени вычислений.")
            
            # Периодически уступаем GIL сетевым потокам
            if i % 10 == 0:
                time.sleep(0.005)
                
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
            raise TimeoutError("Превышен лимит времени вычислений.")
        
        # Периодически уступаем GIL сетевым потокам
        if i % 10 == 0:
            time.sleep(0.005)
            
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
    if processed_img is None: 
        return False
        
    body_mask = np.isnan(processed_img)
    body_ratio = np.sum(body_mask) / body_mask.size
    
    # 1. Базовая проверка площади тела фрактала
    if not (0.015 < body_ratio < 0.65): 
        return False
    
    # 2. Достаточно ли пикселей вне тела для анализа
    non_body = processed_img[~body_mask]
    if non_body.size < 200: 
        return False
    
    # 3. Проверка контрастности (стандартное отклонение)
    std_val = np.std(non_body)
    if std_val < 0.12: 
        return False
    
    # 4. Проверка на богатство цветовых переходов
    unique_vals = np.unique(np.round(non_body, 2))
    if len(unique_vals) < 20: 
        return False
        
    # Подготовка 2D-массива для пространственного анализа (заполняем NaNs нулями)
    clean_2d = np.nan_to_num(processed_img, nan=0.0)
    
    # 5. Детектор шума: Горизонтальная и вертикальная корреляция соседних пикселей.
    # В зашумленных изображениях соседние пиксели практически не коррелируют.
    h_corr = np.corrcoef(clean_2d[:, :-1].flatten(), clean_2d[:, 1:].flatten())[0, 1]
    v_corr = np.corrcoef(clean_2d[:-1, :].flatten(), clean_2d[1:, :].flatten())[0, 1]
    
    if h_corr < 0.45 or v_corr < 0.45:
        # Отклоняем из-за хаотичного высокочастотного шума/зернистости
        return False
        
    # 6. Детектор шума: Нормализованная полная вариация (Total Variation)
    tv_h = np.abs(clean_2d[:, :-1] - clean_2d[:, 1:]).mean() / (std_val + 1e-10)
    tv_v = np.abs(clean_2d[:-1, :] - clean_2d[1:, :]).mean() / (std_val + 1e-10)
    if tv_h > 0.60 or tv_v > 0.60:
        # Слишком резкие перепады яркости на пиксель (шум)
        return False
        
    # 7. Детектор "палок" (диагональных полос / параллельных линий):
    # Применяем быстрое размытие для удаления шума, затем оцениваем сонаправленность градиентов.
    # Если градиенты dY и dX строго линейно зависимы, перед нами монотонные полосы.
    blurred = fast_uniform_filter(clean_2d, size=9)
    dy, dx = np.gradient(blurred)
    
    mag = np.sqrt(dx**2 + dy**2)
    mag_mask = mag > 0.05 * np.max(mag)  # Анализируем только выраженные контуры
    if np.sum(mag_mask) > 100:
        grad_corr = np.abs(np.corrcoef(dx[mag_mask], dy[mag_mask])[0, 1])
        if grad_corr > 0.70:
            # Отклоняем, обнаружена доминирующая структура параллельных линий
            return False
            
    return True

def export_to_buffers_pil(processed_img, cmap=None, target_res=1600):
    if cmap is None:
        cmap = CLASSIC_CMAP
        
    body_mask = np.isnan(processed_img)
    clean_img = np.nan_to_num(processed_img, nan=0.0)
    
    rgba_img = cmap(clean_img)
    rgba_img[body_mask] = [0.0, 0.0, 0.0, 1.0]
    
    rgb_img = (rgba_img[:, :, :3] * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(rgb_img)
    
    if img_pil.width > target_res:
        img_pil = img_pil.resize((target_res, target_res), Image.Resampling.LANCZOS)
        
    buf_jpeg = io.BytesIO()
    img_pil.save(buf_jpeg, format='JPEG', quality=90, optimize=True)
    buf_jpeg.seek(0)
    
    buf_png = io.BytesIO()
    img_pil.save(buf_png, format='PNG')
    buf_png.seek(0)
    
    return buf_jpeg, buf_png

def generate_fractal_pipeline(quality_res=1600, steps=10, progress_callback=None, force_cpu=False, timeout=120.0):
    start_time = time.time()
    deadline = start_time + timeout if timeout else None

    rng = random.Random(secrets.randbits(128))
    seed_int = rng.randint(0, 2**128 - 1)
    
    decoder = EntropyDecoder(seed_int)
    ast_tree = generate_ast(decoder, depth=1, max_depth=4) 
    rpn_tokens = []
    ast_to_rpn(ast_tree, rpn_tokens)
    
    while not validate_rpn(rpn_tokens):
        if time.time() > deadline:
            raise TimeoutError("Превышен таймаут генерации RPN.")
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
    
    if progress_callback:
        progress_callback("⚡ Проверка эстетического потенциала...")
    
    preview_res = 200
    preview_img, _, _ = safe_compute_grid(
        xmin, xmax, ymin, ymax, preview_res, preview_res, final_max_iter, rpn_tokens, is_julia, c_val, use_double=False, deadline=deadline, force_cpu=force_cpu
    )
    preview_processed = apply_adaptive_tonemapping(preview_img, final_max_iter)
    
    if not check_aesthetic_quality(preview_processed):
        log("WARN", "QUALITY", "Фрактал отклонён на стадии превью.")
        return None, None, None, None
        
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
        
    with render_lock:
        final_img, _, _ = safe_compute_grid(
            xmin, xmax, ymin, ymax, render_res, render_res, final_max_iter, rpn_tokens, is_julia, c_val, use_double=True, deadline=deadline, force_cpu=force_cpu
        )
    
    processed_img = apply_adaptive_tonemapping(final_img, final_max_iter)
    
    if not check_aesthetic_quality(processed_img):
        log("WARN", "QUALITY", "Фрактал отклонён финальным фильтром качества.")
        return None, None, None, None
        
    buf_jpeg, buf_png = export_to_buffers_pil(processed_img, CLASSIC_CMAP, target_res=target_res)
    coords_dict = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    
    # Принудительное освобождение тяжелых ресурсов из памяти
    del processed_img
    del final_img
    gc.collect()
    if HAS_TORCH and DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        
    return buf_jpeg, buf_png, formula_str, coords_dict