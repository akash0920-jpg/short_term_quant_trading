## THIS REPOSITORY CONTAINS :
1. bin_class_gem.py : This script , used for extracting features , devicing indicators and traing ML model 
2. final_model.pkl : pickled file of the final trained model
3. trade_exe.py : Loads the pickled model , then run the given csv file(containing the OHLC data of stocks) on the model to stimulate real time trading scenario and finally returns the buy,sell and Hold of different stocks on differet days.
4. Stock price( provided for model training) : this folder contains the dataset provided for model traing and the cleaned datasets used for training and EDA purposes

5. output.png and output2.png :seaborn Pairplot of different features used for training model


# NOTE: 
# 1. SOME OF THE LINES THE SCRIPTS ARE COMMENTED OUT AND ARE EXPECTED TO BE SET ACCORDIG THE END USER , SO PLEASE READ THE COMMENTS CAREFULLY THEY CONATINS INSTRUCTIONS 
# 2. THE STOCK PRICES OF ALL THE STOCKS MUST BE IN THE SAME FOLDER LIKE THE ONE PROVIDED
# 3. MAIN EXECUTABLE IS trade_exe.py , PLEASE SELECT THE TEST FILE LOCATION YOURSELF.