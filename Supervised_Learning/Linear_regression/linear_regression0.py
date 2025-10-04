import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

X = df[['SquareFootage']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# build a smooth 1-D x grid across the whole data range
x_plot = np.linspace(df['SquareFootage'].min(), df['SquareFootage'].max(), 200)

# predict needs 2-D, but plotting wants 1-D
y_line = model.predict(x_plot.reshape(-1, 1)).ravel()

plt.scatter(X_test['SquareFootage'], y_test, label='Actual Data')
plt.plot(x_plot, y_line, label='Regression Line')
plt.xlabel('Square Footage'); plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend(); plt.tight_layout(); plt.show()
