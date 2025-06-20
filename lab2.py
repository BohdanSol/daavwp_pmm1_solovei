import pandas as pd
import numpy as np
import seaborn as sns

# --- Завдання 1 ---
# 1. Створення df1
df1 = pd.DataFrame({
    'ID': range(100, 110),
    'Customer': ['Anna', 'Bohdan', 'Ivan', 'Maria', 'Olga', 'Petro', 'Svitlana', 'Dmytro', 'Kateryna', 'Yurii'],
    'Price': [1200, 350, 499, 1500, 800, 250, 999, 450, 700, 1100],
    'Date': ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05',
             '2024-06-06', '2024-06-07', '2024-06-08', '2024-06-09', '2024-06-10'],
    'Category': ['Electronics', 'Clothing', 'Furniture', 'Electronics', 'Clothing',
                 'Furniture', 'Electronics', 'Clothing', 'Furniture', 'Electronics'],
    'Payment': [True, False, True, True, False, True, False, True, True, False]
})

print("Типи даних до змін:")
print(df1.dtypes)

# 3. Приведення типів
df1['Price'] = df1['Price'].astype('float16')
df1['Date'] = pd.to_datetime(df1['Date'])
df1['Category'] = df1['Category'].astype('category')

print("\nТипи даних після змін:")
print(df1.dtypes)

print("\nВикористання пам'яті до змін:")
df1_before = pd.DataFrame({
    'ID': range(100, 110),
    'Customer': ['Anna', 'Bohdan', 'Ivan', 'Maria', 'Olga', 'Petro', 'Svitlana', 'Dmytro', 'Kateryna', 'Yurii'],
    'Price': [1200, 350, 499, 1500, 800, 250, 999, 450, 700, 1100],
    'Date': ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05',
             '2024-06-06', '2024-06-07', '2024-06-08', '2024-06-09', '2024-06-10'],
    'Category': ['Electronics', 'Clothing', 'Furniture', 'Electronics', 'Clothing',
                 'Furniture', 'Electronics', 'Clothing', 'Furniture', 'Electronics'],
    'Payment': [True, False, True, True, False, True, False, True, True, False]
})
print(df1_before.memory_usage(deep=True))

print("\nВикористання пам'яті після змін:")
print(df1.memory_usage(deep=True))

# --- Завдання 2 ---
# 1. Завантаження даних
df2 = pd.read_csv('london_temp.csv')

# 2. Визначення кількості відсутніх значень у місяці N=8 (серпень)
N = 8
missing_august = df2[df2['month'] == N]['temperature'].isna().sum()
print(f"\nВідсутніх значень у серпні: {missing_august}")

# 3. Замінити відсутні значення у відповідних місяцях
# ((N + 1) % 12) + 1 = 10 (жовтень)
october = ((N + 1) % 12) + 1
df2.loc[(df2['month'] == october) & (df2['temperature'].isna()), 'temperature'] = np.nan

# ((N + 4) % 12) + 1 = 1 (січень)
january = ((N + 4) % 12) + 1
mean_jan = df2[df2['month'] == january]['temperature'].mean()
df2.loc[(df2['month'] == january) & (df2['temperature'].isna()), 'temperature'] = mean_jan

# ((N + 7) % 12) + 1 = 4 (квітень)
april = ((N + 7) % 12) + 1
# Замінити на попередній або наступний день
def fill_prev_next(row):
    if pd.isna(row['temperature']):
        idx = row.name
        prev_temp = df2.loc[idx - 1, 'temperature'] if idx > 0 else np.nan
        next_temp = df2.loc[idx + 1, 'temperature'] if idx < len(df2) - 1 else np.nan
        return prev_temp if not pd.isna(prev_temp) else next_temp
    return row['temperature']

mask_april = (df2['month'] == april) & (df2['temperature'].isna())
df2.loc[mask_april, 'temperature'] = df2[mask_april].apply(fill_prev_next, axis=1)

# 4. Видалити рядки з відсутніми температурами у серпні
df2 = df2[~((df2['month'] == N) & (df2['temperature'].isna()))]

# --- Завдання 3 ---
# 1. Створення фреймів даних
penguins = sns.load_dataset('penguins')
df1_peng = penguins[['species', 'island', 'bill_length_mm', 'bill_depth_mm']]
df2_peng = penguins[['species', 'flipper_length_mm', 'body_mass_g', 'sex']]

# 2. Об'єднання
merged = pd.merge(df1_peng, df2_peng, on='species')
concatenated = pd.concat([df1_peng.set_index('species'), df2_peng.set_index('species')], axis=1, join='inner').reset_index()

# 3. Додаємо size_category
def size_cat(mass):
    return 'small' if mass < 4000 else 'large'
merged['size_category'] = merged['body_mass_g'].apply(size_cat)

# 4. Видалити стовпець sex і рядки з >3 NaN
merged = merged.drop(columns=['sex'])
rows_with_many_na = merged.isna().sum(axis=1) > 3
count_many_na = rows_with_many_na.sum()
merged = merged[~rows_with_many_na]
print(f"\nКількість рядків з більше ніж трьома NaN: {count_many_na}")

# 5. Групування
grouped = merged.groupby('species')[['flipper_length_mm', 'body_mass_g']].mean()
print("\nСередні значення по групах:")
print(grouped)