import os
import json
import time
import random
import secrets
import threading
import numpy as np
import telebot

from config import bot, log, TRANSLATIONS, safe_send_photo, safe_send_document, safe_api_call
from database import stats, load_subscribers, remove_subscriber, get_user_lang, get_user_setting
from math_engine import parse_infix_to_rpn
from renderer import (
    generate_fractal_pipeline, safe_compute_grid, apply_adaptive_tonemapping, 
    export_to_buffers_pil, CLASSIC_CMAP
)

# --- Умный планировщик автоматической рассылки ---
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

def run_broadcast_distribution(epoch):
    subs = load_subscribers()
    if not subs:
        log("INFO", "AUTO", "Подписчиков для рассылки нет.")
        return {"status": "no_subscribers", "sent_count": 0}
        
    pending_users = [cid for cid in subs if not stats.is_broadcast_delivered(epoch, cid)]
    if not pending_users:
        log("INFO", "AUTO", f"Все подписчики уже получили рассылку {epoch}.")
        return {"status": "success", "sent_count": 0}
        
    log("INFO", "AUTO", f"Инициация рассылки для {len(pending_users)} пользователей (эпоха {epoch})...")
    buf_jpeg, buf_png, formula, coords = None, None, None, None
    
    try:
        # 1. Основной цикл генерации
        for attempt in range(1, 16):
            try:
                buf_jpeg, buf_png, formula, coords = generate_fractal_pipeline(
                    quality_res=1600, 
                    steps=10, 
                    force_cpu=False, 
                    timeout=300.0  
                )
                if buf_jpeg is not None:
                    break
            except TimeoutError:
                log("WARN", "AUTO", f"Попытка рассылки {attempt} прервана по таймауту.")
            except Exception as e:
                log("ERROR", "AUTO", f"Сбой пайплайна на попытке {attempt}: {e}")
                
        # 2. Резервный цикл генерации
        if buf_jpeg is None:
            log("WARN", "AUTO", "Запуск резервного режима рассылки с мягкими лимитами...")
            for attempt in range(1, 6):
                try:
                    buf_jpeg, buf_png, formula, coords = generate_fractal_pipeline(
                        quality_res=1200,  
                        steps=5,           
                        force_cpu=False,
                        timeout=180.0
                    )
                    if buf_jpeg is not None:
                        break
                except TimeoutError:
                    pass
                except Exception as e:
                    log("ERROR", "AUTO", f"Сбой в резервном режиме на попытке {attempt}: {e}")
                    
        # 3. Гарантированный спасательный круг (Fallback): красивый кастомный фрагтал Мандельброта
        if buf_jpeg is None:
            log("WARN", "AUTO", "Процедурный поиск не дал результатов. Активация Fallback...")
            try:
                fallback_rpn = parse_infix_to_rpn("(Z^2) + C")
                xmin, xmax, ymin, ymax = -0.7436438870371587, -0.7436438870371587 + 0.00000000005, 0.1318259042053119 - 0.000000000025, 0.1318259042053119 + 0.000000000025
                
                final_img, _, _ = safe_compute_grid(
                    xmin, xmax, ymin, ymax, 1200, 1200, 500, fallback_rpn, False, 0j, use_double=True
                )
                processed_img = apply_adaptive_tonemapping(final_img, 500)
                if processed_img is None:
                    processed_img = np.nan_to_num(final_img)
                    
                buf_jpeg, buf_png = export_to_buffers_pil(processed_img, CLASSIC_CMAP, target_res=1200)
                formula = "(Z^2) + C"
                coords = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
                log("SUCCESS", "AUTO", "Стабильный резервный фрактал успешно материализован.")
            except Exception as fallback_err:
                log("CRITICAL", "AUTO", f"Даже резервный фрактал не удалось построить: {fallback_err}")

        if buf_jpeg is None:
            log("ERROR", "AUTO", "Критический сбой генерации рассылки. Отправка уведомлений о задержке...")
            sent_notif = 0
            for chat_id in pending_users:
                try:
                    user_lang = get_user_setting(chat_id, "lang", None) or get_user_lang(chat_id)
                    if user_lang == "ru":
                        notif_text = (
                            "🌀 **Искажение информационного поля**\n\n"
                            "Наши вычислительные ядра столкнулись с областью сверхвысокой плотности хаоса. "
                            "Материализация планового фрактала задерживается. "
                            "Мы уже пересчитываем сингулярность... Оставайтесь на связи! 🪐"
                        )
                    else:
                        notif_text = (
                            "🌀 **Information Field Distortion**\n\n"
                            "Our computing cores have encountered a region of ultra-high chaos density. "
                            "The materialization of the scheduled fractal is temporarily delayed. "
                            "We are already recalculating the singularity... Stay tuned! 🪐"
                        )
                    safe_api_call(bot.send_message, chat_id, notif_text, parse_mode='Markdown')
                    sent_notif += 1
                except Exception as e:
                    log("ERROR", "AUTO", f"Не удалось отправить уведомление о сбое пользователю {chat_id}: {e}")
                    
            return {"status": "notified_delay", "sent_count": sent_notif}
            
        sent_success = 0
        for chat_id in pending_users:
            try:
                buf_jpeg.seek(0)
                buf_png.seek(0)
                
                user_lang = get_user_setting(chat_id, "lang", None) or get_user_lang(chat_id)
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
                
                stats.record_broadcast_delivery(epoch, chat_id)
                sent_success += 1
                
            except telebot.apihelper.ApiTelegramException as e:
                if e.error_code in [403, 400]:
                    log("WARN", "AUTO", f"Пользователь {chat_id} заблокировал бота. Удаление подписки.")
                    remove_subscriber(chat_id)
                    stats.set_user_inactive(chat_id)
            except Exception as e:
                log("ERROR", "AUTO", f"Ошибка отправки пользователю {chat_id}: {e}")
                
            time.sleep(0.25)
            
        return {"status": "success", "sent_count": sent_success}
        
    finally:
        # Гарантируем закрытие буферов во всех сценариях работы функции
        for b in [buf_jpeg, buf_png]:
            if b is not None:
                try:
                    b.close()
                except Exception:
                    pass

def automated_delivery_loop():
    INTERVAL = 7200
    while True:
        try:
            now = time.time()
            state = load_broadcast_state()
            last_sent = state.get("last_broadcast_epoch", 0.0)
            current_scheduled_slot = (now // INTERVAL) * INTERVAL
            
            if last_sent == 0.0:
                state["last_broadcast_epoch"] = current_scheduled_slot
                save_broadcast_state(state)
                last_sent = current_scheduled_slot
            
            if now >= current_scheduled_slot and last_sent < current_scheduled_slot:
                log("INFO", "AUTO", f"Запуск рассылки за слот {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_scheduled_slot))}")
                run_broadcast_distribution(epoch=current_scheduled_slot)
                state["last_broadcast_epoch"] = current_scheduled_slot
                save_broadcast_state(state)
        except Exception as e:
            log("ERROR", "AUTO", f"Критическая ошибка планировщика рассылки: {e}")
            
        time.sleep(30)