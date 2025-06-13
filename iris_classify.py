import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")
df.drop('Id', axis=1, inplace=True)

X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

def classify_iris():
    print("\n Enter Iris flower measurements to classify the species:")
    try:
        sl = float(input("Sepal Length (cm): "))
        sw = float(input("Sepal Width (cm): "))
        pl = float(input("Petal Length (cm): "))
        pw = float(input("Petal Width (cm): "))
    except ValueError:
        print(" Please enter valid numerical values.")
        return

    user_input = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
    user_input_scaled = scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)
    print(f"\n Predicted Iris Species: **{prediction[0]}**")

classify_iris()