import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Завдання 2 ---
# 1. Попередня обробка даних
peng = pd.read_excel('penguins.xlsx')
print(peng.head())
print(peng.info())

# Перевірка пропущених значень
print(peng.isnull().sum())
peng = peng.dropna()

# Перетворення на категоріальний тип
for col in ['species', 'island', 'sex']:
    peng[col] = peng[col].astype('category')

# 2. Візуалізація даних
sns.set_style('darkgrid')

# Стовпцева діаграма кількості пінгвінів кожного виду за статтю
plt.figure(figsize=(7, 5))
sns.countplot(data=peng, x='species', hue='sex', palette='Set2')
plt.title('Кількість пінгвінів кожного виду за статтю')
plt.xlabel('Вид')
plt.ylabel('Кількість')
plt.legend(title='Стать')
plt.tight_layout()
plt.savefig('penguins_count_species_sex.jpg', dpi=300)
plt.close()

# Стовпцева діаграма медіанної маси тіла за видом та островом
plt.figure(figsize=(7, 5))
sns.barplot(data=peng, x='species', y='body_mass_g', hue='island', estimator='median', errorbar=None, palette='Set1')
plt.title('Медіанна маса тіла пінгвінів за видом та островом')
plt.xlabel('Вид')
plt.ylabel('Медіанна маса тіла (г)')
plt.legend(title='Острів')
plt.tight_layout()
plt.savefig('penguins_median_mass_species_island.jpg', dpi=300)
plt.close()

# Скрипкова діаграма довжини дзьоба за видом та статтю
plt.figure(figsize=(7, 5))
sns.violinplot(data=peng, x='species', y='bill_length_mm', hue='sex', split=True, palette='Set3')
plt.title('Довжина дзьоба пінгвінів за видом та статтю')
plt.xlabel('Вид')
plt.ylabel('Довжина дзьоба (мм)')
plt.legend(title='Стать')
plt.tight_layout()
plt.savefig('penguins_violin_bill_species_sex.jpg', dpi=300)
plt.close()

# Точкова діаграма довжини плавника за видом та островом
plt.figure(figsize=(7, 5))
sns.stripplot(data=peng, x='species', y='flipper_length_mm', hue='island', dodge=True, palette='tab10', jitter=False)
plt.title('Довжина плавника пінгвінів за видом та островом')
plt.xlabel('Вид')
plt.ylabel('Довжина плавника (мм)')
plt.legend(title='Острів')
plt.tight_layout()
plt.savefig('penguins_strip_flipper_species_island.jpg', dpi=300)
plt.close()