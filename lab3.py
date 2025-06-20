import pandas as pd
import numpy as np
import seaborn as sns

# --- Завдання 1 ---
sales = pd.Series([120,135,150,160,145,90,60], index=['Пн','Вт','Ср','Чт','Пт','Сб','Нд'])

# 2. Зростання попиту на 9%
sales_promo = sales * 1.09

# 3. Повернення товару (N = 8)
returns = pd.Series([1,2,3,4,5,6,7], index=sales.index) * 8
sales_final = sales_promo - returns

# 4. Загальна кількість проданих товарів
total_sales = sales_final.sum()

# 5. Середній показник продажу
mean_sales = sales_final.mean()

# 6. Дні з продажами > 130
days_over_130 = sales_final[sales_final > 130].index.tolist()

# 7. Дні з продажами < середнього
days_below_mean = sales_final[sales_final < mean_sales]

# 8. Бінарна маска для 80 <= x <= 120
mask_80_120 = (sales_final >= 80) & (sales_final <= 120)

# 9. Класифікація днів
def classify(x):
    if x > 140:
        return "високі"
    elif x >= 100:
        return "середні"
    else:
        return "низькі"
sales_level = sales_final.apply(classify)

# --- Завдання 2 ---
tips = sns.load_dataset('tips')

# 1. Описові статистики
mean_tip = tips['tip'].mean()
std_tip = tips['tip'].std()
median_total = tips['total_bill'].median()
iqr_total = tips['total_bill'].quantile(0.75) - tips['total_bill'].quantile(0.25)
min_size = tips['size'].min()
max_size = tips['size'].max()
mode_size = tips['size'].mode()[0]

# 2. Групування/агрегація
mean_tip_by_day = tips.groupby('day')['tip'].mean()
mean_tip_ratio = tips.groupby(['sex','smoker']).apply(lambda df: (df['tip']/df['total_bill']).mean())
mean_bill_by_day_time = tips.groupby(['day','time'])['total_bill'].mean()

# 3. Аналіз розподілу
tips['tip_rate'] = tips['tip'] / tips['total_bill']
share_over_15 = (tips['tip_rate'] > 0.15).mean()
tip_rate_quartiles = tips['tip_rate'].quantile([0.25,0.5,0.75])
top5_tip_rate = tips.sort_values('tip_rate', ascending=False).head(5)

# --- Завдання 3 ---
exercs = sns.load_dataset('exercise')

# 1. apply() для pulse_level
def pulse_level(pulse):
    if pulse < 100:
        return 'low'
    elif pulse < 140:
        return 'medium'
    else:
        return 'high'
exercs['pulse_level'] = exercs['pulse'].apply(pulse_level)

# 2. agg() для статистик по kind
pulse_stats = exercs.groupby('kind')['pulse'].agg(['min','mean','max','std'])

# 3. pipe() для середнього пульсу за 30 хв
def avg_pulse_30min(df):
    return df[df['time']=='30 min'].groupby('kind')['pulse'].mean()
avg_pulse_by_kind_30min = exercs.pipe(avg_pulse_30min)

# 4. Середній пульс по id та above_average
mean_pulse_by_id = exercs.groupby('id')['pulse'].mean()
overall_mean_pulse = exercs['pulse'].mean()
above_average = mean_pulse_by_id.apply(lambda x: int(x > overall_mean_pulse))
mean_pulse_by_id = mean_pulse_by_id.to_frame('mean_pulse')
mean_pulse_by_id['above_average'] = above_average

# --- Вивід результатів (для перевірки) ---
if __name__ == "__main__":
    print("Завдання 1:")
    print("sales_final:\n", sales_final)
    print("Загальна кількість:", total_sales)
    print("Середній показник:", mean_sales)
    print("Дні з продажами > 130:", days_over_130)
    print("Дні з продажами < середнього:\n", days_below_mean)
    print("Бінарна маска 80-120:\n", mask_80_120)
    print("Класифікація днів:\n", sales_level)
    print("\nЗавдання 2:")
    print("Середнє tip:", mean_tip, "Стд:", std_tip)
    print("Медіана total_bill:", median_total, "IQR:", iqr_total)
    print("Мін/Макс/Мода size:", min_size, max_size, mode_size)
    print("Середній tip по днях:\n", mean_tip_by_day)
    print("Середнє співвідношення tip/total_bill:\n", mean_tip_ratio)
    print("Середній рахунок по днях/часу:\n", mean_bill_by_day_time)
    print("Частка tip_rate > 15%:", share_over_15)
    print("Квартилі tip_rate:\n", tip_rate_quartiles)
    print("Топ-5 tip_rate:\n", top5_tip_rate[['total_bill','tip','tip_rate']])
    print("\nЗавдання 3:")
    print("pulse_level:\n", exercs[['pulse','pulse_level']].head())
    print("pulse_stats:\n", pulse_stats)
    print("avg_pulse_by_kind_30min:\n", avg_pulse_by_kind_30min)
    print("mean_pulse_by_id:\n", mean_pulse_by_id)