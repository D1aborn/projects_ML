Построение модели машинного обучения для определения месторождений нефти, разработка которых принесет наибольшую прибыль

Вы работаете в добывающей компании «ГлавРосГосНефть». Нужно решить, где бурить новую скважину.

Вам предоставлены пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. Постройте модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализируйте возможную прибыль и риски техникой Bootstrap.

Шаги для выбора локации:

    В избранном регионе ищут месторождения, для каждого определяют значения признаков;
    Строят модель и оценивают объём запасов;
    Выбирают месторождения с самым высокими оценками значений. Количество месторождений зависит от бюджета компании и стоимости разработки одной скважины;
    Прибыль равна суммарной прибыли отобранных месторождений.



<font size="4"><b>План работы</b></font>

    1 Подготовка данных по 3-м регионам;

    2 Обучение и проверка модели для каждого региона:
        - первый регион;
        - второй регион;
        - третий регион;
            
    3 Подготовка к расчёту прибыли:
        - ключевые значения для расчётов в отдельных переменных;
        - рассчет достаточного объёма сырья для безубыточной разработки новой скважины;
        - функция для расчёта прибыли по выбранным скважинам и предсказаниям;
        - подсчет прибыли от разработки выбранных скважин в каждом регионе;
            
    4 Анализ возможной прибыли и рисков техникой Bootstrap для каждого региона:
        - нахождение распределения прибыли в каждом регионе;
        - расчет средней прибыли, 95%-й доверительного интервала и риска убытков для каждого региона;
        
    5 Предложения по выбору региона для разработки скважин
    
    ---
    
    import pandas as pd
import numpy as np
from numpy.random import RandomState
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import r2_score
