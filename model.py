# Выбранная модель в результате исследования
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class Models:
    def __init__(self, model_path: str = "model.pkl"):
        """Инициализация с загрузкой предварительно обученной модели"""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Загружаем модель
        with open(self.model_path, 'rb') as f:
            self.pipe_final = pickle.load(f)

    def model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Получение предсказаний от загруженной модели"""
        # Проверка необходимых колонок
        required_cols = ['gender', 'diet', 'stress_level', 'physical_activity_days_per_week']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные признаки: {', '.join(missing_cols)}")

        try:
            # Получаем предсказания
            proba = pd.DataFrame(
                self.pipe_final.predict_proba(df)[:, 1],
                columns=['proba'],
                index=df.index
            )
            return proba
        except Exception as e:
            raise RuntimeError(f"Ошибка при получении предсказаний: {str(e)}")
