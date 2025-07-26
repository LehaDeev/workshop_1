from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import argparse
import logging
import pandas as pd
from fastapi.staticfiles import StaticFiles
import os

from preprocess import Dataset
from model import Models

app = FastAPI()

# Создаем временную директорию, если ее нет
os.makedirs("tmp", exist_ok=True)
app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

@app.get("/health")
def health():
    return {"status": "Ok"}

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                    {"request": request})

@app.post("/process")
def process_request(file: UploadFile, request: Request):
    try:
        if not file.filename.endswith('.csv'):
            raise ValueError("Пожалуйста, загрузите CSV-файл")

        save_pth = "tmp/" + file.filename
        app_logger.info(f'Обработка файла - {save_pth}')

        # Сохраняем файл
        with open(save_pth, "wb") as fid:
            content = file.file.read()
            if not content:
                raise ValueError("Загружен пустой файл")
            fid.write(content)

        # Читаем данные
        try:
            data = pd.read_csv(save_pth)
            if data.empty:
                raise ValueError("CSV-файл не содержит данных")
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV-файла: {str(e)}")

        # Предобработка данных
        try:
            preprocessor = Dataset(data)
            processed_data = preprocessor.preprocess()
        except ValueError as e:
            # Получаем список столбцов в загруженном файле
            uploaded_columns = ", ".join(data.columns.tolist())
            error_msg = f"{str(e)}\n\nЗагруженные столбцы: {uploaded_columns}"
            raise ValueError(error_msg)

        # Получаем предсказания
        try:
            predictor = Models()
            predictions = predictor.model(processed_data)

            # Форматируем результаты
            results = predictions.reset_index().rename(columns={'index': 'id'})
            results['proba'] = round(results['proba'], 4)#.apply(lambda x: f"{x:.2%}")
            results_dict = results.to_dict(orient='records')

            return templates.TemplateResponse("res_form.html",
                                          {"request": request,
                                           "res": "Success",
                                           "message": "Файл успешно обработан",
                                           "results": results_dict})

        except Exception as e:
            raise ValueError(f"Ошибка предсказания: {str(e)}")

    except ValueError as e:
        app_logger.error(f"Ошибка обработки: {str(e)}")
        return templates.TemplateResponse("error_form.html",
                                      {"request": request,
                                       "error_message": str(e)})

    except Exception as e:
        app_logger.error(f"Неожиданная ошибка: {str(e)}")
        return templates.TemplateResponse("error_form.html",
                                      {"request": request,
                                       "error_message": f"Произошла непредвиденная ошибка: {str(e)}"})


    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
