import pandas as pd
import numpy as np
import os, pickle
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Train dataset will be used for trade execution if Test dataset is not provided
TRAIN_DATA_FOLDER = r"Stock price( provided for model training)"
TEST_DATA_FOLDER = r"--------########---------" # ENTER LOCATION OF UR TEST DATA
OUTPUT_FILE = "Trade_Positions_Log.csv"

# Portfolio tracked in Percentages (100.0 = 100% of Portfolio)
PORTFOLIO_START_PCT = 100.0 
ALLOCATION_PER_TRADE_PCT = 10.0 # Invest 10% of portfolio per trade
MAX_HOLD_DAYS = 3

# ==========================================
# 1. FEATURE ENGINEERING FUNCTION
# ==========================================
def calculate_indicators(df):
    """Calculates all technical indicators required by the model."""
    df = df.copy()
    
    df['SMA_3'] = df['Close'].rolling(window=3).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_Diff'] = df['SMA_3'] - df['SMA_10']
    df['Volatility'] = df['Close'].rolling(window=5).std()
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=(14 - 1), min_periods=14).mean()
    avg_loss = loss.ewm(com=(14 - 1), min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Fill NAs backward to prevent dropping too much data during warm-up
    df = df.bfill()
    return df

# ==========================================
# 2. LOADING THE MODEL
# ==========================================
features = ['SMA_3', 'SMA_10', 'MA_Diff', 'Volatility', 'RSI', 'Daily_Return']
# Ensure your model file is in the same directory or provide full path
model = pickle.load(open('final_model.pkl', 'rb'))

# ==========================================
# 3. THE TRADING ENGINE (STATE MACHINE)
# ==========================================
print("\nRunning Trading Engine on Test Data...")
current_portfolio_pct = PORTFOLIO_START_PCT

# State Machine tracking for each stock
positions = {
    stock_id: {
        'state': 'FLAT',          # 'FLAT' or 'LONG'
        'buy_day': None, 
        'buy_price': 0, 
        'days_held': 0, 
        'pending_action': None    # 'BUY' or 'SELL' triggered from yesterday's close
    } for stock_id in range(1, 11)
}

completed_trades = []

# Load test data and prep indicators
test_frames = {}
for stock_id in range(1, 11):
    file_path = os.path.join(TEST_DATA_FOLDER, f"S{stock_id}_Stock_Prices.xlsx")
    if not os.path.exists(file_path):
        file_path = os.path.join(TRAIN_DATA_FOLDER, f"S{stock_id}_Stock_Prices.xlsx") 
        
    df = pd.read_excel(file_path)
    df.columns = ['Day', 'Close', 'High', 'Low', 'Open']
    df = calculate_indicators(df)
    test_frames[stock_id] = df

# Find the maximum number of days in the test set to simulate a daily loop
max_days = max([len(df) for df in test_frames.values()])

for day_index in range(max_days):
    for stock_id in range(1, 11):
        if day_index >= len(test_frames[stock_id]):
            continue 
            
        today_data = test_frames[stock_id].iloc[day_index]
        today_day = today_data['Day']
        today_open = today_data['Open']
        today_close = today_data['Close']
        
        pos = positions[stock_id]
        
        # ---------------------------------------------------------
        # PHASE A: Execute Pending Actions at TODAY'S OPEN
        # ---------------------------------------------------------
        if pos['pending_action'] == 'BUY':
            pos['state'] = 'LONG'
            pos['buy_day'] = today_day
            pos['buy_price'] = today_open
            pos['days_held'] = 0
            pos['pending_action'] = None
            
        elif pos['pending_action'] == 'SELL':
            sell_day = today_day
            sell_price = today_open
            
            # Calculate % difference and update portfolio
            pct_diff = (sell_price - pos['buy_price']) / pos['buy_price']
            profit_loss = ALLOCATION_PER_TRADE_PCT * pct_diff
            current_portfolio_pct += profit_loss
            
            # Log Trade
            completed_trades.append({
                'Stock_Name': f"S{stock_id}",
                'Day_of_Buying': pos['buy_day'],
                'Buy_Price': round(pos['buy_price'], 2),
                'Day_of_Selling': sell_day,
                'Sell_Price': round(sell_price, 2),
                'Pct_Portfolio_Invested': ALLOCATION_PER_TRADE_PCT,
                'Pct_Difference': round(pct_diff * 100, 2) # Formatted as percentage
            })
            
            pos['state'] = 'FLAT'
            pos['buy_day'] = None
            pos['buy_price'] = 0
            pos['days_held'] = 0
            pos['pending_action'] = None

        # ---------------------------------------------------------
        # PHASE B: Check Max Hold Rule at TODAY'S CLOSE
        # ---------------------------------------------------------
        if pos['state'] == 'LONG':
            pos['days_held'] += 1
            
            # If we hit the 3-day max, we MUST square off at today's close
            if pos['days_held'] == MAX_HOLD_DAYS:
                sell_day = today_day
                sell_price = today_close # Squared off at Close as per rules
                
                pct_diff = (sell_price - pos['buy_price']) / pos['buy_price']
                profit_loss = ALLOCATION_PER_TRADE_PCT * pct_diff
                current_portfolio_pct += profit_loss
                
                completed_trades.append({
                    'Stock_Name': f"S{stock_id}",
                    'Day_of_Buying': pos['buy_day'],
                    'Buy_Price': round(pos['buy_price'], 2),
                    'Day_of_Selling': sell_day,
                    'Sell_Price': round(sell_price, 2),
                    'Pct_Portfolio_Invested': ALLOCATION_PER_TRADE_PCT,
                    'Pct_Difference': round(pct_diff * 100, 2)
                })
                
                pos['state'] = 'FLAT'
                pos['buy_day'] = None
                pos['buy_price'] = 0
                pos['days_held'] = 0
                pos['pending_action'] = None

        # ---------------------------------------------------------
        # PHASE C: Generate New Signals for Tomorrow
        # ---------------------------------------------------------
        # Only predict if we don't already have an action pending
        if pos['pending_action'] is None:
            X_today = today_data[features].values.reshape(1, -1)
            signal = model.predict(X_today)[0]
            
            if pos['state'] == 'FLAT' and signal == 1:
                pos['pending_action'] = 'BUY'
                
            elif pos['state'] == 'LONG' and signal == 0:
                pos['pending_action'] = 'SELL'

# ==========================================
# 4. EXPORT RESULTS
# ==========================================
log_df = pd.DataFrame(completed_trades)
log_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSimulation Complete!")
print(f"Starting Portfolio Value: {PORTFOLIO_START_PCT}%")
print(f"Final Portfolio Value: {current_portfolio_pct:.2f}%")
print(f"Total Portfolio Return: {current_portfolio_pct - PORTFOLIO_START_PCT:.2f}%")
print(f"Total Trades Executed: {len(log_df)}")
print(f"Detailed trade log saved to: {OUTPUT_FILE}")