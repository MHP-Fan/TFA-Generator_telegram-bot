import threading
import signal
import sys
import time
import random
import secrets      # <-- Добавлено!
import re
import io
import numpy as np  # <-- Добавлено!
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import telebot      # <-- Добавлено!
from telebot import types
from dotenv import load_dotenv

load_dotenv()

# Импорты конфигурации
from config import (
    bot, BOT_TOKEN, ADMIN_ID, log, TRANSLATIONS, safe_api_call, safe_send_photo,
    safe_send_document, safe_edit_message_text, ProgressUpdater, bot_phrases,
    get_random_phrase, DEFAULT_PHRASES_EN, HAS_TORCH, DEVICE,
    render_lock     # <-- Добавлено!
)

# Импорты баз данных
from database import (
    stats, user_manager, load_subscribers, save_subscriber, remove_subscriber,
    save_user_setting, get_user_setting, get_user_lang
)

# Импорты математики
from math_engine import parse_infix_to_rpn, validate_rpn

# Импорты рендеринга
from renderer import (
    safe_compute_grid, find_boundary_point_v2, apply_adaptive_tonemapping,
    export_to_buffers_pil, CLASSIC_CMAP, generate_fractal_pipeline
)

# Импорты планировщика
from scheduler import automated_delivery_loop, run_broadcast_distribution

# Клавиатура
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

# --- Handlers ---
# Перехват сообщений в режиме офлайн (для всех, кроме администратора)
@bot.message_handler(func=lambda message: stats.get_system_state("status", "online") == "offline" and message.chat.id != ADMIN_ID)
def handle_offline_mode(message):
    try:
        lang = get_user_lang(message)
        if lang == "ru":
            msg_text = "⚙️ *Бот временно отключен администратором на техническое обслуживание.*"
        else:
            msg_text = "⚙️ *The bot is temporarily disabled by the administrator for technical maintenance.*"
        safe_api_call(bot.send_message, message.chat.id, msg_text, parse_mode='Markdown')
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Ошибка отправки сообщения об офлайне: {e}")

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
        lang = new_lang
        
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
    
    lines = text.split("\n")
    for line in lines:
        if "Формула:" in line or "Formula:" in line:
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
        
    buf_jpeg, buf_png = None, None # Вынесено на уровень выше для finally
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
        
        with render_lock:
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
        
        stats.log_generation(chat_id, "custom", final_max_iter)
    except Exception as e:
        log("ERROR", "CUSTOM_RENDER", f"Ошибка при кастомном рендере: {e}")
        safe_api_call(bot.send_message, chat_id, t["gen_error"].format(error=str(e)), reply_markup=get_main_keyboard(chat_id, lang))
    finally:
        # Гарантируем освобождение ОЗУ
        for b in [buf_jpeg, buf_png]:
            if b is not None:
                try: b.close()
                except Exception: pass
        user_manager.end_job(chat_id)


# В функции send_fractal:
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

    buf_jpeg, buf_png = None, None # Вынесено на уровень выше для finally
    try:
        random_steps = random.randint(steps_min, steps_max)
        status_msg = safe_api_call(bot.send_message, chat_id, mode_text.format(steps=random_steps))
        updater = ProgressUpdater(bot, chat_id, status_msg.message_id)
        
        formula, coords = None, None
        max_attempts = 15
        
        for attempt in range(1, max_attempts + 1):
            def make_callback(att):
                return lambda text: updater.update(f"{t['attempt'].format(attempt=att, max_attempts=max_attempts)}\n└ {text}")
            
            updater.update(t["init_grid"].format(attempt=attempt, max_attempts=max_attempts), force=True)
            
            try:
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
                log("ERROR", "TIMEOUT", f"Таймаут на попытке {attempt}: {te}")
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
        
        buf_png.name = f"fractal_{secrets.token_hex(4)}.png"
        safe_send_document(
            chat_id,
            buf_png,
            caption=t["png_caption"],
            parse_mode='Markdown'
        )
        
        stats.log_generation(chat_id, gen_type, random_steps)
        
    except TimeoutError:
        log("ERROR", "TELEGRAM", "Генерация прервана по таймауту.")
        try:
            safe_api_call(bot.send_message, chat_id, t["timeout_error"], reply_markup=get_main_keyboard(chat_id, lang))
        except Exception:
            pass
    except telebot.apihelper.ApiTelegramException as te:
        if te.error_code in [403, 400]:
            remove_subscriber(chat_id)
            stats.set_user_inactive(chat_id)
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Сбой отправки: {e}")
        try:
            safe_api_call(bot.send_message, chat_id, t["gen_error"].format(error=str(e)), reply_markup=get_main_keyboard(chat_id, lang))
        except Exception:
            pass
    finally:
        # Гарантируем закрытие буферов во всех сценариях работы функции
        for b in [buf_jpeg, buf_png]:
            if b is not None:
                try: b.close()
                except Exception: pass
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
        try: safe_api_call(bot.send_message, chat_id, t["busy"])
        except Exception: pass
        return
    elif status == "cooldown":
        try: safe_api_call(bot.send_message, chat_id, t["cooldown"].format(val=val))
        except Exception: pass
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
        log("INFO", "TELEGRAM", f"Пользователь {chat_id} (@{username}) запросил пакет из {num} фракталов.")
        
        updater = ProgressUpdater(bot, chat_id, status_msg.message_id)
        generated = 0
        
        for i in range(num):
            buf_jpeg, buf_png, formula, coords = None, None, None, None
            max_attempts = 15
            
            for attempt in range(1, max_attempts + 1):
                random_steps = random.randint(steps_min, steps_max)
                
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
                    t["batch_step"].format(index=i+1, num=num, attempt=attempt, max_attempts=max_attempts, text=t["vectors"]),
                    force=True
                )
                
                try:
                    buf_jpeg, buf_png, formula, coords = generate_fractal_pipeline(
                        quality_res=1600, 
                        steps=random_steps, 
                        progress_callback=make_batch_callback(i, attempt)
                    )
                except TimeoutError:
                    log("WARN", "TIMEOUT", f"Пакет: Фрактал {i+1} прерван по таймауту.")
                    if attempt == max_attempts:
                        break
                    updater.update(t["timeout_retry"].format(attempt=attempt, max_attempts=max_attempts), force=True)
                    time.sleep(1.0)
                    continue
                
                if buf_jpeg is not None:
                    break
                else:
                    updater.update(
                        t["batch_step"].format(index=i+1, num=num, attempt=attempt, max_attempts=max_attempts, text=t["unstable_chaos"]),
                        force=True
                    )
                    time.sleep(1.0)
            
            if buf_jpeg is not None:
                try:
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
                    generated += 1
                except telebot.apihelper.ApiTelegramException as te:
                    if te.error_code in [403, 400]:
                        remove_subscriber(chat_id)
                        stats.set_user_inactive(chat_id)
                        break  
                except Exception as e:
                    log("ERROR", "TELEGRAM", f"Ошибка отправки кадра {i+1}: {e}")
                finally:
                    buf_jpeg.close()
                    buf_png.close()
                time.sleep(1.0)
                
        try: bot.delete_message(chat_id, status_msg.message_id)
        except Exception: pass
            
        if generated == 0:
            safe_api_call(bot.send_message, chat_id, t["batch_failed"], reply_markup=get_main_keyboard(chat_id, lang))
        else:
            safe_api_call(bot.send_message, chat_id, t["batch_success"], reply_markup=get_main_keyboard(chat_id, lang))
            stats.log_generation(chat_id, f"batch_{zoom_mode}", num)
    except Exception as e:
        log("ERROR", "TELEGRAM", f"Ошибка пакетной генерации {chat_id}: {e}")
        try: safe_api_call(bot.send_message, chat_id, t["gen_error"].format(error=str(e)), reply_markup=get_main_keyboard(chat_id, lang))
        except Exception: pass
    finally:
        user_manager.end_job(chat_id)

# --- Админ панель ---
@bot.message_handler(commands=['admin_shutdown', 'admin_offline'])
def handle_admin_shutdown(message):
    if message.chat.id != ADMIN_ID:
        return 
        
    try:
        log("WARN", "SYSTEM", f"Администратор {message.chat.id} инициировал отключение бота.")
        safe_api_call(bot.send_message, ADMIN_ID, "⚠️ *Инициировано принудительное отключение бота...*", parse_mode='Markdown')
        
        # 1. Установка статуса 'offline' в базе данных
        stats.set_system_state("status", "offline")
        
        # 2. Обновление короткого описания в Telegram
        try:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/setMyShortDescription",
                json={"short_description": "🔴 Вне сети. Сервер временно отключен на техническое обслуживание."},
                timeout=5
            )
            log("SUCCESS", "SYSTEM", "Описание бота успешно изменено на 'Офлайн' в Telegram API.")
        except Exception as e:
            log("ERROR", "SYSTEM", f"Не удалось обновить статус короткого описания: {e}")
            
        # 3. Отправка финального сообщения администратору
        safe_api_call(bot.send_message, ADMIN_ID, "✅ *Бот переведен в статус OFFLINE. Процесс завершается.*", parse_mode='Markdown')
        
        # 4. Сброс логов на диск
        sys.stdout.flush()
        sys.stderr.flush()
        
        # 5. Остановка пуллинга и завершение работы
        bot.stop_polling()
        sys.exit(0)
    except Exception as e:
        log("ERROR", "SYSTEM", f"Ошибка при выполнении отключения бота: {e}")
        safe_api_call(bot.send_message, ADMIN_ID, f"❌ Ошибка отключения: {e}")

@bot.message_handler(commands=['admin_stats'])
def send_admin_report(message):
    if message.chat.id != ADMIN_ID:
        return 
        
    report = stats.get_weekly_report()
    text = (
        "📊 **Фрактальный Навигатор: Аналитика**\n\n"
        f"👥 Всего пользователей: `{report['total_users']}`\n"
        f"🟢 Активных сессий: `{report['active_users']}`\n"
        f"🌀 Всего генераций: `{report['total_generations']}`\n\n"
    )
    for g_type, count in report['gen_distribution'].items():
        text += f"• `{g_type}`: {count}\n"
        
    dates = [item[0] for item in report['user_growth_7d']]
    counts = [item[1] for item in report['user_growth_7d']]
    
    if dates:
        plt.figure(figsize=(6, 4))
        plt.plot(dates, counts, marker='o', color='#2269eb', linewidth=2)
        plt.title("Рост аудитории", fontsize=10)
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
        safe_api_call(bot.send_message, ADMIN_ID, text + "\n_Мало данных для графика._", parse_mode='Markdown')

def safe_run_manual_broadcast():
    try:
        manual_epoch = int(time.time())
        report = run_broadcast_distribution(epoch=manual_epoch)
        if report["status"] == "success":
            msg = f"✅ **Ручная рассылка завершена!**\n\n• Отправлено: `{report['sent_count']}`"
        elif report["status"] == "notified_delay":
            msg = f"⚠️ **Сбой!**\n\n• Разосланы уведомления: `{report['sent_count']}`"
        else:
            msg = f"⚠️ **Статус:** `{report['status']}`"
        safe_api_call(bot.send_message, ADMIN_ID, msg, parse_mode='Markdown')
    except Exception as e:
        log("ERROR", "ADMIN", f"Сбой при выполнении ручной рассылки: {e}")

@bot.message_handler(commands=['admin_broadcast'])
def trigger_manual_broadcast(message):
    if message.chat.id != ADMIN_ID:
        return 
    safe_api_call(bot.send_message, ADMIN_ID, "⏳ Инициализирована ручная рассылка...")
    thread = threading.Thread(target=safe_run_manual_broadcast, name="ManualBroadcast", daemon=True)
    thread.start()

# Пинг для сервера-наблюдателя
def heartbeat_loop():
    while True:
        try:
            requests.get("https://MHPFan.pythonanywhere.com/ping", timeout=30)
        except Exception as e:
            log("WARN", "SYSTEM", f"Не удалось отправить пинг: {e}")
        time.sleep(60)

# --- Инициализация и запуск процесса ---
if __name__ == "__main__":
    delivery_thread = threading.Thread(target=automated_delivery_loop, name="AutoSend", daemon=True)
    delivery_thread.start()
    
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/setMyShortDescription",
            json={"short_description": "🟢 В сети. Нажмите /start, чтобы раствориться в бездне."},
            timeout=10
        )
        stats.set_system_state("status", "online")
        log("INFO", "SYSTEM", "Статус бота успешно переведен в 'Онлайн'.")
    except Exception as e:
        log("ERROR", "SYSTEM", f"Не удалось обновить статус: {e}")

    ping_thread = threading.Thread(target=heartbeat_loop, name="Heartbeat", daemon=True)
    ping_thread.start()
    
    log("INFO", "SYSTEM", "Бот успешно запущен в режиме ожидания...")
    
    try:
        while True:
            try:
                bot.infinity_polling(timeout=90, long_polling_timeout=30)
            except Exception as e:
                log("ERROR", "SYSTEM", f"Сбой пуллинга: {e}. Перезапуск через 5 секунд...")
                time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        log("INFO", "SYSTEM", "Получен сигнал остановки процесса.")
    finally:
        try: signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception: pass
        log("INFO", "SYSTEM", "Перевожу статус бота в режим 'Офлайн'...")
        try:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/setMyShortDescription",
                json={"short_description": "🔴 Вне сети. Сервер временно отключен на техническое обслуживание."},
                timeout=5
            )
            log("SUCCESS", "SYSTEM", "Статус офлайна успешно установлен. Процесс завершен.")
        except Exception as e:
            log("ERROR", "SYSTEM", f"Не удалось обновить статус перед выходом: {e}")
        
        # Гарантируем сброс всех буферов логов на диск
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass