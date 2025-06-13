import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('car data.csv')
print("\n First 5 rows of the dataset:")
print(df.head())

print("\n Columns in the dataset:", df.columns.tolist())

le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le_seller.fit_transform(df['Selling_type'])   
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

df.drop('Car_Name', axis=1, inplace=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

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
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.show()

def predict_price():
    print("\n Enter details to predict car selling price:")

    try:
        year = int(input("Model Year (e.g. 2015): "))
        present_price = float(input("Present Price (in lakhs): "))
        kms_driven = int(input("Kilometers Driven: "))
        fuel_type_input = input("Fuel Type (Petrol/Diesel/CNG): ").capitalize()
        selling_type_input = input("Selling Type (Dealer/Individual): ").capitalize()
        transmission_input = input("Transmission Type (Manual/Automatic): ").capitalize()
        owner = int(input("Number of Previous Owners (0/1/3): "))
    except ValueError:
        print(" Invalid input. Please enter valid numeric values where applicable.")
        return

    try:
        fuel_type_encoded = le_fuel.transform([fuel_type_input])[0]
        selling_type_encoded = le_seller.transform([selling_type_input])[0]
        trans_type_encoded = le_trans.transform([transmission_input])[0]
    except:
        print(" Invalid category input. Please match exactly with training labels (e.g. 'Petrol', 'Dealer', 'Manual').")
        return

    user_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type_encoded,
                               selling_type_encoded, trans_type_encoded, owner]],
                             columns=X.columns)

    predicted_price = model.predict(user_data)[0]
    print(f"\n Estimated Selling Price: ₹ {predicted_price:.2f} lakhs")

predict_price()
