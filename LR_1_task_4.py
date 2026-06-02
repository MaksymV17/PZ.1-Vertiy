import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

# ==============================================================================
# 1. СТВОРЕННЯ ДАНИХ (Генерація замість читання з файлу data_multivar_nb.txt)
# ==============================================================================
# Генеруємо 400 точок, розбитих на 4 класи, як у типовому файлі 'data_multivar_nb.txt'
X, y = make_blobs(n_samples=400, n_features=2, centers=4, cluster_std=1.2, random_state=42)

# Розбиваємо дані на навчальну (train) та тестову (test) вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# ==============================================================================
# 2. ФУНКЦІЯ ВІЗУАЛІЗАЦІЇ (Заміна файлу utilities)
# ==============================================================================
def visualize_classifier(classifier, X, y, title="Класифікатор"):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    mesh_step_size = 0.02
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), 
                                     np.arange(y_min, y_max, mesh_step_size))
    
    output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    output = output.reshape(x_values.shape)
    
    plt.pcolormesh(x_values, y_values, output, cmap=plt.cm.Paired, shading='auto', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', linewidth=1, cmap=plt.cm.Paired, s=40)
    
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.title(title)
    plt.xlabel("Ознака 1")
    plt.ylabel("Ознака 2")
    plt.show()


# ==============================================================================
# 3. СТВОРЕННЯ ТА НАВЧАННЯ НАЇВНОГО БАЙЄСІВСЬКОГО КЛАСИФІКАТОРА
# ==============================================================================
# Ініціалізація моделі Gaussian Naive Bayes
classifier = GaussianNB()

# Навчання моделі на тренувальних даних
classifier.fit(X_train, y_train)

# Перевірка точності моделі на тестових даних
accuracy = classifier.score(X_test, y_test)
print(f"Точність класифікатора на тестовій вибірці: {accuracy * 100:.2f}%")

# Візуалізація результатів для навчальної вибірки
visualize_classifier(classifier, X_train, y_train, "Найвний Байєс — Навчальна вибірка")

# Візуалізація результатів для тестової вибірки
visualize_classifier(classifier, X_test, y_test, "Найвний Байєс — Тестова вибірка")