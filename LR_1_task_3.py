import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# ==============================================================================
# ФУНКЦІЯ ВІЗУАЛІЗАЦІЇ
# ==============================================================================
def visualize_classifier(classifier, X, y):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    mesh_step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), 
                                     np.arange(y_min, y_max, mesh_step_size))
    
    output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    output = output.reshape(x_values.shape)
    
    plt.pcolormesh(x_values, y_values, output, cmap=plt.cm.Paired, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', linewidth=1, cmap=plt.cm.Paired, s=60)
    
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.title("Логістичний класифікатор (Виправлено)")
    plt.xlabel("Ознака X1")
    plt.ylabel("Ознака X2")
    plt.show()

# ==============================================================================
# ОСНОВНИЙ КОД ЗІ СКРІНШОТУ
# ==============================================================================

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5],
              [6, 5], [5.6, 5], [3.3, 0.4],
              [3.9, 0.9], [2.8, 1],
              [0.5, 3.4], [1, 4], [0.6, 4.9]])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# ВИПРАВЛЕНО: замість 'liblinear' використовуємо сучасний 'lbfgs', який підтримує багато класів
classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)

# Навчання класифікатора (тепер помилки не буде)
classifier.fit(X, y)

# Візуалізація результатів роботи
visualize_classifier(classifier, X, y)