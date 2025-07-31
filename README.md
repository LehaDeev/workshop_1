# Прогнозирование риска сердечного приступа

## Описание
Веб-приложение для оценки риска сердечного приступа на основе медицинских данных

## Установка

1. Создать виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```bash

2. Установить зависимости:
```bash
pip install -r requirements.txt
```bash
3. Запустить приложение:
```bash
astapi dev main.py
```bash
Использование
1. Откройте
```bash
 http://localhost:8000
```bash
2. Загрузите CSV-файл с медицинскими данными

3. Получите прогноз риска для каждого пациента
