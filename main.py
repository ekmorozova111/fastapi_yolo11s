# Импортируем библиотеки 

# Импортируем библиотеку для передачи изображения внутри JSON-ответа.
import base64
# Импортируем библиотеку для сохранения сложных данных в SQLite и чтения истории из базы данных
import json
# Импортируем библиотеку для создания базы данных и хранения в ней истории всех запросов. 
import sqlite3
# Импортируем библиотеку для измерения времени работы моей модели, чтобы рассчитать скорость обработки запроса.
import time
# Импортируем библиотека для быстрого расчета сложной статистики по всем запросам. С её помощью вычисляются средние значения и квантили
import numpy as np
# Здесь мы превращаем скачанные байты изображения в объект, который «понимает» библиотеку для работы с картинками (PIL).
from io import BytesIO
#  Импортируем библиотеку для работы с графическими данными, чтобы узнать ширину и высоту загруженного изображения.
from PIL import Image
# Здесь Эта строка импортирует инструменты для создания веб-интерфейса сервера, обработки загрузки файлов и управления ошибками при передаче данных.
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
# Здесь мы импортируем инструмент для управления жизненным циклом приложения
from contextlib import asynccontextmanager
# Импортируем класс из инференс.ру для работы с моделью, который отвечает за загрузку весов модели и запуск детекции
from inference import ModelHandler

# Укажем путь к базе данных 
DB_PATH = "history.db"

# создадим базу данных
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Добавляем колонки для времени обработки и размеров
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                process_time REAL,
                width INTEGER,
                height INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

# Сделаем обращение модели к классу из файла инференса 
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.model = ModelHandler("yolo11s_best.pt")
    yield

app = FastAPI(lifespan=lifespan)

# Здесь сделаем код ошибки 400 с текстом ‘bad request’
@app.post("/forward")
# поправила обращение, а то не работал запрос загрузки изображения
async def forward(image: UploadFile = File(...), request: Request = None):
    if image is None or not image.filename:
        raise HTTPException(status_code=400, detail="bad request")

    start_time = time.perf_counter()
    try:
        image_bytes = await image.read()
        
        # Получаем размеры изображения
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size

        # Инференс
        results = request.app.state.model.predict(image_bytes)
        
        process_time = time.perf_counter() - start_time

        # Сохраняем расширенную историю
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO history (filename, process_time, width, height) VALUES (?, ?, ?, ?)",
                (image.filename, process_time, width, height)
            )
        # Теперь вернем код ошибки 403 и сообщение: “модель не смогла обработать данные”
        return {
            "status": "success",
            "image_base64": base64.b64encode(image_bytes).decode('utf-8'),
            "detections": results
        }
    except Exception as e:
        raise HTTPException(status_code=403, detail="модель не смогла обработать данные")

#  Делаем запрос хистори
@app.get("/history")
async def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM history").fetchall()
        return [dict(row) for row in rows]

# Делаемзапрос со статистикой 
@app.get("/stats")
async def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT process_time, width, height FROM history")
        data = cursor.fetchall()
    
    if not data:
        return {"message": "нет доступных данных"}

    times = [row[0] for row in data]
    widths = [row[1] for row in data]
    heights = [row[2] for row in data]

    return {
        "время_обработки": {
            "среднее": float(np.mean(times)),
            "50%": float(np.percentile(times, 50)),
            "95%": float(np.percentile(times, 95)),
            "99%": float(np.percentile(times, 99))
        },
        "характеристики_изображений": {
            "средняя_ширина": float(np.mean(widths)),
            "средняя_высота": float(np.mean(heights)),
            "максимальная_ширина": int(np.max(widths)),
            "максимальная_высота": int(np.max(heights))
        },
        "всего_запросов": len(times)
    }
