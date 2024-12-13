from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from keras.models import load_model
import numpy as np

# Инициализация FastAPI
app = FastAPI()

# Настройка шаблонов Jinja2 для рендеринга HTML
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

# Загрузка модели
model = load_model("iris_model.h5")

# Определение классов ирисов
iris_classes = ["Setosa", "Versicolor", "Virginica"]


# Маршрут для отображения формы
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


# Маршрут для обработки данных из формы
@app.post("/predict/")
async def predict(
        request: Request,
        sepal_length: float = Form(...),  # Длина чашелистика
        sepal_width: float = Form(...),  # Ширина чашелистика
        petal_length: float = Form(...),  # Длина лепестка
        petal_width: float = Form(...)  # Ширина лепестка
):
    # Преобразование данных в массив numpy
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Предсказание модели
    predictions = model.predict(input_data)
    predicted_class = iris_classes[np.argmax(predictions)]

    # Возвращение результата пользователю
    return templates.TemplateResponse("result.html", {"request": request, "prediction": predicted_class})