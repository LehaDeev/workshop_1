import pandas as pd
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, data: pd.DataFrame):
        """Инициализация с преобразованием названий столбцов и валидацией"""
        self.required_features = [
            'age', 'gender', 'bmi',
            'cholesterol',
            'triglycerides', 'ck_mb', 'troponin',
            'diabetes', 'family_history', 'smoking', 'obesity',
            'alcohol_consumption', 'previous_heart_problems',
            'medication_use', 'diet', 'stress_level',
            'physical_activity_days_per_week'
        ]

        # Сначала преобразуем названия столбцов
        self.data = self._clean_column_names(data.copy())

        # Затем выполняем валидацию
        self._validate_input(self.data)

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приведение названий столбцов к единому стилю"""
        df.columns = (
            df.columns
            .str.lower()  # в нижний регистр
            .str.strip()  # убираем пробелы по краям
            .str.replace(r'[:()\s\-/]', '_', regex=True)  # заменяем спецсимволы на подчеркивание
            .str.replace(r'_+', '_', regex=True)  # заменяем множественные подчеркивания на одно
            .str.rstrip('_')  # убираем подчеркивание в конце
        )
        return df

    def _validate_input(self, df: pd.DataFrame):
        """Валидация входных данных после преобразования названий столбцов"""
        missing = [req_col for req_col in self.required_features
                  if req_col not in df.columns]

        if missing:
            raise ValueError(
                f"Отсутствуют обязательные признаки: {', '.join(missing)}\n"
                f"Имеющиеся столбцы: {', '.join(df.columns)}"
            )

    def _convert_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Преобразование указанных столбцов в целые числа"""
        int_columns = [
            'diabetes', 'family_history', 'smoking', 'obesity',
            'alcohol_consumption', 'previous_heart_problems',
            'medication_use', 'stress_level',
            'physical_activity_days_per_week'
        ]

        for col in int_columns:
            if col in df.columns:
                # Безопасное преобразование
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
        return df

    def _process_biomarkers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка биомаркеров и создание бинарных признаков"""
        # Обработка CK-MB
        if 'ck_mb' in df.columns:
            df['ck_mb'] = pd.to_numeric(df['ck_mb'], errors='coerce')
            df['risk_ck_mb'] = (df['ck_mb'] >= 0.05).astype(int)
            df = df.drop('ck_mb', axis=1, errors='ignore')

        # Обработка тропонина
        if 'troponin' in df.columns:
            df['troponin'] = pd.to_numeric(df['troponin'], errors='coerce')
            df['risk_troponin'] = (df['troponin'] >= 0.04).astype(int)
            df = df.drop('troponin', axis=1, errors='ignore')

        return df

    def preprocess(self) -> pd.DataFrame:
        """Основной метод предобработки данных"""
        try:
            df = self.data.copy()

            # Удаление ненужных столбцов
            df = df.drop(columns=['unnamed'], errors='ignore')

            # Заменим индексы на id пациентов
            df = df.set_index('id')

            # Удаление строк с пропущенными значениями
            df = df.dropna()

            # Преобразование типов
            df = self._convert_to_int(df)

            # Обработка биомаркеров
            df = self._process_biomarkers(df)

            # Проверка, что остались данные после предобработки
            if df.empty:
                raise ValueError("После предобработки данных не осталось - проверьте входные данные")

            return df

        except Exception as e:
            logger.error(f"Ошибка предобработки данных: {str(e)}")
            raise ValueError(f"Ошибка предобработки данных: {str(e)}")
