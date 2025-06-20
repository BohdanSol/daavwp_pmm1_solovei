import pandas as pd
import numpy as np

# 1. Завантаження даних
df = pd.read_csv('london_temp.csv', index_col=0)

# 2. Повна інформація про фрейм даних
print("Інформація про DataFrame:")
print(df.info())
print(df.describe())

# 3. Перші N рядків
N = 8

# 4. Температури для трьох місяців певної пори року
season_months = {
    'зима': ['Dec', 'Jan', 'Feb'],
    'весна': ['Mar', 'Apr', 'May'],
    'літо': ['Jun', 'Jul', 'Aug'],
    'осінь': ['Sep', 'Oct', 'Nov']
}
season = input("Введіть пору року (зима, весна, літо, осінь): ").strip().lower()
months = season_months.get(season)
if months:
    print(f"Температури для {season}:")
    print(df[months])
else:
    print("Невірна пора року.")

# 5. Температури другої декади (10-19 дні) всіх місяців
print("Температури другої декади (10-19 дні) всіх місяців:")
print(df.loc[10:19])

# 6. Температури для введених місяців
months_input = input("Введіть перелік місяців через кому (наприклад, Jan,Mar,May): ")
months_list = [m.strip() for m in months_input.split(',')]
print(df[months_list])

# 7. Температури для введених днів
days_input = input("Введіть перелік днів через кому (наприклад, 1,5,10): ")
days_list = [int(d.strip()) for d in days_input.split(',')]
print(df.loc[days_list])

# 8. Середньомісячна температура у місяці N
month_num = int(input("Введіть номер місяця (1-12): "))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_col = month_names[month_num-1]
mean_temp = df[month_col].mean(skipna=True)
print(f"Середньомісячна температура у {month_col}: {mean_temp:.2f}")

# 9. Замінити NaN на середньомісячну температуру у місяці N
df[month_col] = df[month_col].fillna(mean_temp)
print(f"Після заміни NaN у {month_col}:")
print(df[month_col])

# 10. Кількість днів з температурою у діапазоні [a, b] у місяці N
a = float(input("Введіть нижню межу температури: "))
b = float(input("Введіть верхню межу температури: "))
count_in_range = df[(df[month_col] >= a) & (df[month_col] <= b)][month_col].count()
print(f"Кількість днів у {month_col} з температурою в діапазоні [{a}, {b}]: {count_in_range}")

# --- Завдання 3 ---
# 1. Вибір стовпця-місяця як Series
sr = df[month_col]

# 2. Мінімальна і максимальна температура
print(f"Мінімальна температура у {month_col}: {sr.min()}")
print(f"Максимальна температура у {month_col}: {sr.max()}")

# 3. Середня і медіанна температура
print(f"Середня температура у {month_col}: {sr.mean():.2f}")
print(f"Медіанна температура у {month_col}: {sr.median()}")

# 4. Останні 7 температур після сортування за спаданням
print("Останні 7 температур (за спаданням):")
print(sr.sort_values(ascending=False).tail(7))

# 5. Кількість і відсоток днів з температурою не нижче введеного значення
t = float(input("Введіть порогову температуру: "))
count_above = sr[sr >= t].count()
percent_above = 100 * count_above / sr.count()
print(f"Кількість днів з температурою не нижче {t}: {count_above}")
print(f"Відсоток таких днів: {percent_above:.2f}%")