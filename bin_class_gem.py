import pandas as pd
import numpy as np
import os,pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the folder where your 10 Excel files are located
folder_path = r"D:\DataSets\Art of compounding\Stock price"
all_stocks_data = []

# --- 1. DATA PREPARATION (DAILY LOGIC) ---
for stock_id in range(1, 11):
    file_name = f"S{stock_id}_Stock_Prices.xlsx"
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        continue
        
    df = pd.read_excel(file_path)
    df.columns = ['Day', 'Close', 'High', 'Low', 'Open']
    
    # 1. Technical Indicators (Features)
    df['SMA_3'] = df['Close'].rolling(window=3).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_Diff'] = df['SMA_3'] - df['SMA_10']
    
    # Rolling Volatility (e.g., standard deviation of last 5 days)
    df['Volatility'] = df['Close'].rolling(window=5).std()
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=(14 - 1), min_periods=14).mean()
    avg_loss = loss.ewm(com=(14 - 1), min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Daily Return (Momentum proxy)
    df['Daily_Return'] = df['Close'].pct_change()

    # 2. THE BINARY TARGET (Will it go up tomorrow?)
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df['Target'] = np.where(df['Next_Day_Close'] > df['Close'], 1, 0)
    
    # Drop rows with NaNs caused by rolling windows and shifts
    df = df.dropna()
    df['Stock'] = stock_id
    
    all_stocks_data.append(df)

# Combine into one continuous dataset
final_df = pd.concat(all_stocks_data, ignore_index=True)

# Save the new binary dataset (Optional)
# final_df.to_csv(os.path.join(folder_path, "Daily_Binary_Stock_Data.csv"), index=False)


# --- 2. MODEL TRAINING (BINARY CLASSIFICATION) ---
features = ['SMA_3', 'SMA_10', 'MA_Diff', 'Volatility', 'RSI', 'Daily_Return']
X = final_df[features]
y = final_df['Target']

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Binary Classifier
# Setting class_weight='balanced' helps if there are slightly more down days than up days
rf = RandomForestClassifier(n_estimators=300, 
    max_depth=5, 
    min_samples_split=20, 
    min_samples_leaf=10, 
    max_features='sqrt',
    class_weight='balanced', 
    random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Total Daily Datapoints: {len(final_df)}")
print(f"Binary Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

importances = rf.feature_importances_
print("\nFeature Importances:")
for col, imp in zip(features, importances):
    print(f"{col}: {imp:.4f}")

# saving The model in Local folder
# pickle.dump(rf,open('final_model.pkl','wb'))