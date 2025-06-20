import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Завдання 1 ---
# 1. Попередня обробка даних
diam = pd.read_csv('DA&V_lab-05_diamonds.tsv', sep='\t')
print(diam.head())
print(diam.info())

# Перевірка пропущених значень
print(diam.isnull().sum())
diam = diam.dropna()

# Перетворення на категоріальний тип
for col in ['cut', 'color', 'clarity']:
    diam[col] = diam[col].astype('category')

# 2. Візуалізація даних
sns.set_style('whitegrid')

# KDE ціни за cut
plt.figure(figsize=(8, 5))
sns.kdeplot(data=diam, x='price', hue='cut', fill=True, common_norm=False, palette='Set2')
plt.title('KDE розподілу ціни діамантів за якістю огранки')
plt.xlabel('Ціна')
plt.ylabel('Щільність')
plt.legend(title='Огранка')
plt.tight_layout()
plt.savefig('diamonds_kde_price_cut.png', dpi=300)
plt.close()

# ECDF carat за color
plt.figure(figsize=(8, 5))
sns.ecdfplot(data=diam, x='carat', hue='color', palette='Set1')
plt.title('ECDF ваги діамантів за кольором')
plt.xlabel('Вага (carat)')
plt.ylabel('ECDF')
plt.legend(title='Колір')
plt.tight_layout()
plt.savefig('diamonds_ecdf_carat_color.png', dpi=300)
plt.close()

# Гістограма + KDE depth за clarity
plt.figure(figsize=(8, 5))
sns.histplot(data=diam, x='depth', hue='clarity', element='step', stat='density', common_norm=False, palette='tab10', bins=30)
sns.kdeplot(data=diam, x='depth', hue='clarity', common_norm=False, palette='tab10', fill=False, legend=False)
plt.title('Глибина діамантів за чистотою')
plt.xlabel('Глибина (%)')
plt.ylabel('Щільність')
plt.legend(title='Чистота')
plt.tight_layout()
plt.savefig('diamonds_hist_kde_depth_clarity.png', dpi=300)
plt.close()

