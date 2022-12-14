#### Анализ продаж компьютерных игр и выявление закономерностей, определяющих их успешность

---

Интернет-магазин продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы. Необходимо провести анализ данных и выявить закономерности, определяющие успешность игр. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.

В наборе данных попадается аббревиатура ESRB (Entertainment Software Rating Board) — это ассоциация, определяющая возрастной рейтинг компьютерных игр. ESRB оценивает игровой контент и присваивает ему подходящую возрастную категорию, например, «Для взрослых», «Для детей младшего возраста» или «Для подростков».

 *Статус проекта - завершен.*
 
---

**Ход работы**

1. Обзор данных.

2. Подготовка данных: 
    - замена названий столбцов;
    - преобразование данных в нужные типы;
    - обработка пропусков.

3. Исследовательский анализ данных:
    - определение количества игр, выпущенных в разные годы. Важны ли данные за все периоды?
    - анализ изменений продаж по платформам. Выбор платформы с наибольшими суммарными продажами и построение распределения по годам. За какой характерный срок появляются новые и исчезают старые платформы?
    - определение платформ лидирующих по продажам, расту или падению. Выбор нескольких потенциально прибыльных платформ.
    - построение графика «ящик с усами» по глобальным продажам игр в разбивке по платформам. Описание результата.
    - анализ влияния на продажи внутри одной популярной платформы отзывы пользователей и критиков. Построение диаграммы рассеяния и подсчет корреляции между отзывами и продажами.
    - соотнесение выводов с продажами игр на других платформах.
    - анализ распределения игр по жанрам. Самые прибыльные жанры. Жанры с высокими и низкими продажами.

4. Составление портрета пользователя каждого региона:
    - самые популярные платформы (топ-5), различия в долях продаж;
    - самые популярные жанры (топ-5), разница;
    - влияние рейтинга ESRB на продажи в отдельном регионе.

5. Проверка гипотез:
    - средние пользовательские рейтинги платформ Xbox One и PC одинаковые;
    - средние пользовательские рейтинги жанров Action и Sports разные.

6. Общий вывод.

---

**Стек технологий проекта:**

- pandas
- math
- numpy
- matplotlib
- scipy 
- copy
- seaborn

---

**Для просмотра проекта:**
 - используйте файл: *Analysis_sales_computer_games.ipynb*;
 - или пройдите по ссылке: https://github.com/D1aborn/projects_ML/blob/main/Analysis_sales_computer_games/Analysis_sales_computer_games.ipynb
