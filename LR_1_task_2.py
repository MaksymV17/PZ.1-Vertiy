import numpy as np
from sklearn import preprocessing

# ==============================================================================
# 1. ВХІДНІ ДАНІ ДЛЯ ВАРІАНТУ №5
# ==============================================================================
# Беремо 12 чисел з таблиці для 5-го варіанту
raw_data = np.array([-1.3, 3.9, 4.5, -5.3, -4.2, -1.3, 5.2, -6.5, -1.1, -5.2, 2.6, -2.2])

# Перетворюємо в матрицю 4 рядки на 3 колонки, щоб працювали всі попередні методи
input_data = raw_data.reshape(4, 3)

# Поріг бінаризації для 5-го варіанту з таблиці
binarization_threshold = 3.0

print("=== [Варіант 5] Початкова матриця даних ===")
print(input_data)
print("-" * 50)


# ==============================================================================
# 2.1 БІНАРИЗАЦІЯ ДАНИХ (з індивідуальним порогом 3.0)
# ==============================================================================
data_binarized = preprocessing.Binarizer(threshold=binarization_threshold).transform(input_data)
print(f"\n=== 2.1 Бінаризовані дані (Поріг = {binarization_threshold}) ===")
print(data_binarized)


# ==============================================================================
# 2.2 ВИЛУЧЕННЯ СЕРЕДНЬОГО (СТАНДАРТИЗАЦІЯ)
# ==============================================================================
print("\n=== 2.2 Вилучення середнього ===")
print("BEFORE Mean =", input_data.mean(axis=0).round(4))
print("BEFORE Std deviation =", input_data.std(axis=0).round(4))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER Mean (має бути ~0) =", data_scaled.mean(axis=0).round(4))
print("AFTER Std deviation (має бути 1) =", data_scaled.std(axis=0).round(4))


# ==============================================================================
# 2.3 МАСШТАБУВАННЯ ОЗНАК (Min-Max)
# ==============================================================================
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\n=== 2.3 Масштабування ознак у діапазон [0, 1] ===")
print(data_scaled_minmax)


# ==============================================================================
# 2.4 НОРМАЛІЗАЦІЯ ДАНИХ (L1 та L2)
# ==============================================================================
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')

print("\n=== 2.4 L1-Нормалізовані дані ===")
print(data_normalized_l1.round(4))
print("\n=== 2.4 L2-Нормалізовані дані ===")
print(data_normalized_l2.round(4))