# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load dataset
df = pd.read_csv("house_prices.csv", sep=",", engine="python")
# Fix cases where data might be read as one long column
if len(df.columns) == 1:
    df = df[df.columns[0]].str.split(",", expand=True)
    df.columns = ["Size (sqft)", "Bedrooms", "Age (years)", "Price (RM)"]
    df = df.apply(pd.to_numeric)



# Step 3: Explore data
print(df.describe())
print(df.head())

# Optional: Visualize relationships
plt.figure(figsize=(12, 4))
for i, col in enumerate(["Size (sqft)", "Bedrooms", "Age (years)"]):
    plt.subplot(1, 3, i + 1)
    plt.scatter(df[col], df["Price (RM)"])
    plt.xlabel(col)
    plt.ylabel("Price (RM)")
    plt.title(f"{col} vs Price")

plt.tight_layout()
plt.show()

# Step 4: Prepare features and target
X = df[["Size (sqft)", "Bedrooms", "Age (years)"]]
y = df["Price (RM)"]

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Print model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 9: Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (RM)")

plt.ylabel("Predicted Price (RM)")
plt.title("Actual vs Predicted Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.grid(True)
plt.show()