import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Loading Dataset

sys.stdout.reconfigure(encoding='utf-8')
data = pd.read_csv("SeoulBikeData.csv", encoding="cp949")

print("Seoul Bike dataset successfully loaded!")
print(f"\nDATASET SIZE: {data.shape[0]} Rows, {data.shape[1]} Columns")

print("\nFIRST 10 RECORDS:")
print(data.head(10))

print("\nCOLUMN NAMES:")
print(data.columns)

print("\nDATASET INFORMATION:")
data.info()

print("\nMISSING VALUES:")
print(data.isnull().sum())

# 2. Cleaning Dataset & Logging
print("\nMISSING VALUES BEFORE CLEANING:")
print(data.isnull().sum())

print("\nDUPLICATE ROWS:", data.duplicated().sum())

# Drop entirely empty columns if present
empty_columns = [c for c in data.columns if data[c].isnull().all()]
data = data.drop(columns=empty_columns)

print(f"\nSHAPE AFTER CLEANING: {data.shape}")

# 3. Exploratory Data Analysis (EDA)

print("\nDESCRIPTIVE STATISTICS:")
pd.set_option('display.max_columns', None)
print(data.describe())

# A. Distribution of bike rental counts
plt.figure(figsize=(8, 5))
plt.hist(data["Rented Bike Count"], bins=30, edgecolor="black")
plt.title("Distribution of Bike Rental Counts")
plt.xlabel("Rented Bike Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# B. Average rentals by hour
hourly_rentals = data.groupby("Hour")["Rented Bike Count"].mean()
plt.figure(figsize=(10, 5))
plt.bar(hourly_rentals.index, hourly_rentals.values)
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Rented Bikes")
plt.xticks(range(24))
plt.tight_layout()
plt.show()

# C. Average rentals by season
seasonal_rentals = data.groupby("Seasons")["Rented Bike Count"].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
seasonal_rentals.plot(kind="bar")
plt.title("Average Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Average Rented Bikes")
plt.tight_layout()
plt.show()

# D. Relationship between Temperature and Bike Rental Count

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Temperature(캜)"],
    data["Rented Bike Count"],
    alpha=0.3
)

plt.title("Relationship between Temperature and Bike Rental Count")
plt.xlabel("Temperature")
plt.ylabel("Rented Bike Count")

plt.tight_layout()
plt.show()  

# 4. Feature Preparation & Engineering
processed_data = data.copy()

# Datetime conversion
processed_data["Date"] = pd.to_datetime(processed_data["Date"], dayfirst=True, errors="coerce")
processed_data["Month"] = processed_data["Date"].dt.month.astype(str)
processed_data["Day of Week"] = processed_data["Date"].dt.day_of_week.astype(str)
processed_data["Hour"] = processed_data["Hour"].astype(str)

# Select features
X = processed_data.drop(columns=["Rented Bike Count", "Date"])
y_continuous = processed_data["Rented Bike Count"]

# One-hot encode categorical features
X = pd.get_dummies(X, dtype=int)

# 5. Data Splitting
X_train, X_test, y_train_continuous, y_test_continuous = train_test_split(
    X, y_continuous,
    test_size=0.2,
    random_state=42,
)


# Calculate the target median using ONLY the isolated training split data
train_median = y_train_continuous.median()
print(f"\nNon-Stratified Training Median Threshold: {train_median}")


# Binarize both targets using the training set's threshold
y_train = (y_train_continuous >= train_median).astype(int)
y_test = (y_test_continuous >= train_median).astype(int)


# Correlation Matrix
correlation_data = X_train.copy()
correlation_data["Demand Level"] = y_train

# Calculate correlation with Demand Level
correlation = correlation_data.corr(numeric_only=True)["Demand Level"]
correlation = correlation.drop(["Demand Level"], errors="ignore").sort_values(ascending=False)

# Visualize Correlation
plt.figure(figsize=(10, 8))
plt.barh(correlation.index, correlation.values)
plt.gca().invert_yaxis()  # Highest correlation at top
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Correlation of Input Features with Demand Level")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Input Features")
plt.tight_layout()
plt.show()



# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Baseline Model
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train_scaled, y_train)
dummy_pred = dummy_model.predict(X_test_scaled)
dummy_acc = accuracy_score(y_test, dummy_pred)

print(f"\n--- BASELINE MODEL EVALUATION ---")
print(f"Baseline Accuracy: {dummy_acc * 100:.2f}%")

# Plot Baseline Confusion Matrix
dummy_cm = confusion_matrix(y_test, dummy_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(dummy_cm, display_labels=["Low Demand", "High Demand"]).plot(
    cmap="Blues", ax=ax, colorbar=False
)
plt.title("Baseline Model Confusion Matrix")
plt.tight_layout()
plt.show()

# 8. Model Training - Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 9. Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- LOGISTIC REGRESSION EVALUATION ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Improvement over Baseline: {(accuracy - dummy_acc) * 100:.2f}%")

# Plot Logistic Regression Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, 
    display_labels=["Low Demand", "High Demand"], 
    cmap="Blues"
)
plt.title("Confusion Matrix - Bike Rental Demand")
plt.tight_layout()
plt.show()

# Classification Report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=["Low Demand", "High Demand"]))