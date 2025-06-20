import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

# --- Завдання 3 ---
# 1. Попередня обробка даних
# Підключення до бази даних та імпорт таблиці mpg
conn = sqlite3.connect('mpg.sql')
mpg = pd.read_sql('SELECT * FROM mpg', conn)
conn.close()

print(mpg.head())
print(mpg.info())

# Перевірка пропущених значень
print(mpg.isnull().sum())
mpg = mpg.dropna()

# Перетворення на категоріальний тип
for col in ['origin', 'cylinders', 'model_year']:
    mpg[col] = mpg[col].astype('category')

# 2. Візуалізація даних
sns.set_style('darkgrid')

# Діаграма розсіювання mpg ~ horsepower за origin
plt.figure(figsize=(8, 5))
sns.scatterplot(data=mpg, x='horsepower', y='mpg', hue='origin', palette='Set1')
plt.title('Витрати пального (mpg) залежно від потужності двигуна за країною-виробником')
plt.xlabel('Потужність (horsepower)')
plt.ylabel('Витрати пального (mpg)')
plt.legend(title='Країна')
plt.tight_layout()
plt.savefig('mpg_scatter_hp_origin.jpg', dpi=300)
plt.close()

# Лінія регресії mpg ~ weight
plt.figure(figsize=(8, 5))
sns.regplot(data=mpg, x='weight', y='mpg', scatter_kws={'s': 20}, line_kws={'color': 'red'})
plt.title('Регресія: Витрати пального (mpg) ~ Вага авто')
plt.xlabel('Вага (weight)')
plt.ylabel('Витрати пального (mpg)')
plt.tight_layout()
plt.savefig('mpg_reg_weight.jpg', dpi=300)
plt.close()

# Лінії регресії mpg ~ displacement за cylinders
plt.figure(figsize=(8, 5))
sns.lmplot(data=mpg, x='displacement', y='mpg', hue='cylinders', palette='tab10', aspect=1.3, height=5, ci=None)
plt.title('Регресія: Витрати пального (mpg) ~ Об\'єм двигуна за кількістю циліндрів')
plt.xlabel('Об\'єм двигуна (displacement)')
plt.ylabel('Витрати пального (mpg)')
plt.tight_layout()
plt.savefig('mpg_reg_displacement_cylinders.jpg', dpi=300)
plt.close()

# Матриця парних діаграм для mpg, horsepower, weight, displacement за origin
sns.pairplot(mpg, vars=['mpg', 'horsepower', 'weight', 'displacement'], hue='origin', palette='Set2', diag_kind='kde')
plt.suptitle('Парні діаграми для основних характеристик авто', y=1.02)
plt.tight_layout()
plt.savefig('mpg_pairplot_origin.jpg', dpi=300)
plt.close()

# Парний графік mpg ~ acceleration з розподілами вздовж осей
plt.figure(figsize=(8, 5))
sns.jointplot(data=mpg, x='acceleration', y='mpg', kind='scatter', marginal_kws=dict(bins=20, fill=True))
plt.suptitle('Витрати пального (mpg) залежно від часу розгону (acceleration)', y=1.02)
plt.tight_layout()
plt.savefig('mpg_jointplot_acceleration.jpg', dpi=300)
plt.close()