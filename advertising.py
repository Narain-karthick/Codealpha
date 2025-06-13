import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("Advertising.csv")

print("\n First 5 rows of the dataset:")
print(df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Any missing values?\n", df.isnull().sum())

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n R² Score: {r2:.3f}")
print(f" Mean Squared Error: {mse:.3f}")


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal Fit")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales\nR² = {r2:.2f}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

coeffs = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\n Impact of Ad Channels on Sales:")
print(coeffs)

def predict_sales():
    print("\n Predict Sales Based on Ad Spend")
    try:
        tv = float(input("TV Advertising Spend: "))
        radio = float(input("Radio Advertising Spend: "))
        newspaper = float(input("Newspaper Advertising Spend: "))
    except ValueError:
        print(" Please enter valid numerical values.")
        return

    sample = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    predicted_sales = model.predict(sample)[0]
    print(f"\n Predicted Sales: {predicted_sales:.2f} units")

predict_sales()
