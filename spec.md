# ARCHITECTURAL BLUEPRINT: FRACTAL NAVIGATOR BOT

## 1. Global Infrastructure & State
* **Execution Devices (`config`):** `HAS_TORCH` (bool), `DEVICE` (cuda/cpu). CUDA is pre-warmed on `MainThread` to prevent deadlocks.
* **Threading Locks:** 
  * `log_lock`: Thread-safe console output. Unbuffered flushing (`sys.stdout.flush()`) is guaranteed during graceful teardown or process signals.
  * `render_lock`: GPU calculation barrier (prevents concurrent heavy PyTorch jobs).
  * `subscribers_lock` & `settings_lock` & `broadcast_lock`: File I/O protection.
* **Storage Entities:**
  * `bot_stats.db` (SQLite): Managed via thread-safe `StatsManager` (internal connection lock).
  * `subscribers.txt`: Plain text, set of active `chat_id`s.
  * `user_settings.json`: Map of `{str(chat_id): {"lang": "ru"|"en", "zoom_mode": "shallow"|"deep"}}`.
  * `broadcast_state.json`: Track scheduling epoch via `{"last_broadcast_epoch": float}`.

## 2. Database Schema (SQLite)
* `users`: `chat_id` (PK, INT), `join_date` (TEXT), `is_active` (INT, default 1).
* `generations`: `id` (AI, PK), `chat_id` (INT), `timestamp` (TEXT), `gen_type` (TEXT), `steps` (INT).
* `subscriptions`: `id` (AI, PK), `chat_id` (INT), `timestamp` (TEXT), `action` (TEXT).
* `broadcast_deliveries`: `epoch` (INT), `chat_id` (INT), `timestamp` (TEXT) -> Composite PK `(epoch, chat_id)` to prevent duplicates.

## 3. Math & Parser Engine (RPN)
* **Tokens:** Variables (`VAR_Z=0`, `VAR_C=1`), Constants (`CONST=2`), Unary (`OP_SIN=8`, `OP_COS=9`... up to `OP_SIGM=15`), Binary (`OP_ADD=3`... `OP_POW=7`).
* **Parser (`parse_infix_to_rpn(expr_str)`):** Uses Shunting-yard algorithm. Tokenizes inputs, handles complex numbers (`X+Yj`), handles unary minus.
* **AST Generator (`generate_ast` & `EntropyDecoder`):** Procedural tree generation with max depth of 4 to avoid chaotic divergence.
* **Evaluation (`evaluate_rpn_pytorch` / `evaluate_rpn_numpy`):** Stack-based. Uses clamping on real/imaginary parts (e.g., `-15.0 to 15.0` for `exp`/`sin`/`pow`) to avoid NaN/Infinity. Adds `EPS_REG = 1e-20` to prevent division by zero.

## 4. Rendering & Search Pipeline
* **Grid Computing (`safe_compute_grid`):** Meshgrid initialization -> iteration loop. Implements double exit condition: escape radius (`mag_sq > 1e8`) and attractor trap (`distance < 1e-12`). Returns raw iteration map. Supports strict computation deadline (`time.time() > deadline`). Yields thread execution control to the event loop every 10 iterations (`time.sleep(0.005)`) to prevent Global Interpreter Lock (GIL) starvation and ensure uninterrupted Telegram network polling.
* **Boundary Positioning (`find_boundary_point_v2`):** Compute gradient of iterations -> apply `fast_uniform_filter` (box blur 15x15) to detect boundary transitions -> return weighted random coordinates.
* **Aesthetic Filter (`check_aesthetic_quality`):** 
  * **Adaptive Tone-mapping:** Dynamic gamma correction based on iteration median.
  * **Basic Color Checks:** Body ratio (black space) must be $0.015 < ratio < 0.65$; Standard deviation of non-body pixels must be $>0.12$; Number of unique colors must be $>20$.
  * **Noise Detector (Spatial Correlation & TV):** Calculates horizontal/vertical adjacent pixel correlation (must be $>0.45$ to reject salt-and-pepper grain) and normalized Total Variation (must be $<0.60$ to reject chaotic high-frequency noise).
  * **Stripe / Parallel Lines Detector:** Applies `fast_uniform_filter` (box blur 9x9) to eliminate micro-noise, calculates 2D Sobel/central gradients, and computes the Pearson correlation of the gradients (`dx` and `dy`). A gradient correlation $>0.70$ flags and rejects concentric parallel stripes ("sticks").
* **Export & Resource Conservation (`export_to_buffers_pil`):** Colormap mapping using a refined palette (with an elevated luminance floor of `#070f2b` and `#111f4d` for low iterations to keep dark outer fractal boundaries visible against the black background) -> SSAA downsampling (Lanczos) -> returns `BytesIO(JPEG)` (90% quality) and `BytesIO(PNG)` (lossless). Explicit memory garbage collection (`gc.collect()`) and CUDA cache eviction (`torch.cuda.empty_cache()`) are strictly executed immediately after exporting to prevent OOM termination.

## 5. Main Execution Flows

### A. Manual Fractal Generation
```
User Request -> Cooldown/Concurrency Check (UserManager) -> Random Steps Selection 
  -> Loop (up to 15 attempts):
       Generate AST/RPN -> Apply boundary zoom (X steps) -> Quick Render Preview (200x200) 
       -> Aesthetic Filter Pass? 
            YES -> Final Render (1600x1600 / SSAA) -> Export Buffers -> Send JPEG & PNG -> End Job
            NO  -> Log Rejection -> Next Attempt
```

### B. Auto-Scheduler & Broadcast (`automated_delivery_loop`)
* Runs in background thread. Interval: 7200 seconds (2 hours).
* Compares current epoch with `last_broadcast_epoch`.
* If true, triggers `run_broadcast_distribution(epoch)`:
  * Generates a high-quality fractal (max 15 attempts, fallback to simplified parameters on failure, ultimate fallback to beautiful Mandelbrot coordinates if all else fails).
  * Distributes to subscribers sequentially. Marks delivery in `broadcast_deliveries` (blocks duplicates). Automatically unsubscribes users who blocked the bot (API 403/400).