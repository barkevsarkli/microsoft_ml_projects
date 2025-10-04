import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Salary_dataset.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

sorted_idx = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color="lightblue", label="Training data", alpha=0.6)
plt.scatter(X_test, y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, y_pred_sorted, color="red", linewidth=2, label="Regression line")

plt.title("Salary vs Years of Experience", fontsize=14)
plt.xlabel("Years of Experience", fontsize=12)
plt.ylabel("Salary", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
