Построение модели оптимизации производства, на примере золотодобывающего предприятия

Подготовьте прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.

Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки.

Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.


## План работы

1 Обзор и подготовка данных

    1.1 Обзор данных и изучение общей информации (3 таблицы);
    1.2 Проверка правильности расчета эффективности обогащения;
    1.3 Анализ признаков, недоступных в тестовой выборке;
    1.4 Предобработка данных

2 Анализ данных

    2.1  Анализ изменения концентрация металлов (Au, Ag, Pb) на различных этапах очистки;
    2.2  Сравнение распределения размеров гранул сырья на обучающей и тестовой выборках.
    2.3  Исследуем суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.

3  Построение модели

    3.1  Напишем функцию для вычисления итоговой sMAPE
    3.2  Обучим разные модели и оценим их качество кросс-валидацией. Выберем лучшую модель и проверим её на тестовой         выборке.

4 Общий вывод


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats as st
import copy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
