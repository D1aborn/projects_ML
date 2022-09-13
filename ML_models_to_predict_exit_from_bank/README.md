Построение различных моделей машинного обучения для прогнозирования ухода клиентов из банка, повышение качества моделей

Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.

Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком.

Постройте модель с предельно большим значением F1-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте F1-меру на тестовой выборке самостоятельно.

Дополнительно измеряйте AUC-ROC, сравнивайте её значение с F1-мерой.


<font size="4"><b>План работы</b></font>

    1 Подготовка данных

    2 Исследование задачи
        2.1 Подготовим данные для проведения иссделования
            - проведем порядковое кодирование столбцов с категориальными признаками
            - проведем масштабирование (стандартизацию) признаков
        2.2 Разобъем данные на 3 выборки: обучающую, валидационную и тестовую и исследуем баланс классов
            Проведем масштабирование (стандартизацию) признаков 3-х выборок
        2.3 Проведем обучение моделей без учета дисбаланса классов и найдем для каждой модели значение метрик F1 и AUC-ROC
            - Решающее дерево
            - Случайный лес
            - Логистическая регрессия
            
    3 Борьба с дисбалансом
        3.1 Сбалансируем классы с помощью операции 'upsampling'
        3.2 Проведем обучение моделей с учетом баланса классов и найдем для каждой модели значение метрик F1 и AUC-ROC
            - Решающее дерево
            - Случайный лес
            - Логистическая регрессия
            
    4 Тестирование модели

    5 Общий вывод
    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
