import pandas as pd
import seaborn as sns

# --- Завдання 1 ---
# 1. Завантаження даних
tit = sns.load_dataset("titanic")

# 1.1 Інформація про змінну class
print("Тип даних:", tit['class'].dtype)
print("Категорії:", tit['class'].unique())

# Перетворення на категорійну змінну з порядком
tit['class'] = pd.Categorical(
    tit['class'],
    categories=['First', 'Second', 'Third'],
    ordered=True
)
print("Коди категорій:", tit['class'].cat.codes)
print("Порядковість:", tit['class'].cat.ordered)

# 2. Таблиця кількості пасажирів у кожній категорії
print(tit['class'].value_counts())

# 3. Перейменування категорій
tit['class'] = tit['class'].cat.rename_categories({'First': '1st', 'Second': '2nd', 'Third': '3rd'})
print(tit['class'].unique())

# 4. Об'єднання категорій 2nd та 3rd у others
tit['class'] = tit['class'].replace({'2nd': 'others', '3rd': 'others'})
tit['class'] = pd.Categorical(tit['class'])
print(tit['class'].unique())

# 5. Очистити список категорій
tit['class'] = tit['class'].cat.remove_unused_categories()
print(tit['class'].cat.categories)

# 6. Впорядкувати категорії: others < 1st
tit['class'] = tit['class'].cat.reorder_categories(['others', '1st'], ordered=True)
tit_sorted = tit.sort_values('class')
print(tit_sorted[['class']].head())

# --- Завдання 2 ---
# 1. Описові статистики для age
print("Описові статистики для age (загалом):")
print(tit['age'].describe())

print("Описові статистики для age за статтю:")
print(tit.groupby('sex')['age'].describe())

# 2. Порівняння середнього віку та медіани
mean_age = tit.groupby('sex')['age'].mean()
median_age = tit.groupby('sex')['age'].median()
print("Середній вік:", mean_age)
print("Медіана віку:", median_age)
if mean_age['male'] > mean_age['female']:
    print("В середньому чоловіки були старші.")
else:
    print("В середньому жінки були старші.")

# 3. Описові статистики для alive
print("alive (загалом):")
print(tit['alive'].value_counts())

print("alive за категорією who:")
print(tit.groupby('who')['alive'].value_counts())

# 4. Аналіз виживших/загиблих
print("Вижило/загинуло (загалом):")
print(tit['alive'].value_counts())

print("Вижило/загинуло за pclass:")
print(tit.groupby('pclass')['alive'].value_counts())

# --- Завдання 3 ---
# 1. crosstab: кількість пасажирів за sex та class
ct = pd.crosstab(tit['sex'], tit['class'], margins=True)
print("Crosstab:\n", ct)

# 2. pivot_table: середній вік за sex та class
pt_age = tit.pivot_table(values='age', index='sex', columns='class', aggfunc='mean')
print("Pivot table (age):\n", pt_age)

# 3. pivot_table: середня вартість квитка за embarked та class
pt_fare = tit.pivot_table(values='fare', index='embarked', columns='class', aggfunc='mean').round(2)
print("Pivot table (fare):\n", pt_fare)

# 4. Порівняння методів
print("""
crosstab зручно використовувати для підрахунку кількостей (категоріальні змінні).
pivot_table — для обчислення агрегованих статистик (наприклад, середнє, сума) для числових змінних у розрізі категорій.
""")

