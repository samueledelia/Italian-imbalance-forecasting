import sys

import tensorflow
print(sys.executable)

import os
os.chdir(os.path.join(os.getcwd(), 'notebook'))
print("Current Working Directory after change:", os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt

# Add the source directory to the system path
sys.path.append(os.path.abspath('../src'))
from open_data import fetch_db_table_sqlserver16
import utils
import importlib

import time
import warnings
import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt




# Reload the module
importlib.reload(utils)
while True:
    # SQL query to fetch data
    sql_query = "SELECT * FROM TERNA_SUNSET_SEGNO_QH" 
    qh = fetch_db_table_sqlserver16(sql=sql_query, verbose=False)
    qh = qh.sort_values(by='ORAINI')
    # Filter the DataFrame for the relevant columns and rows
    qh = qh[['MACROZONA', 'ORAINI', 'SBIL_MWH']]
    # Convert ORAINI to datetime using the correct method
    qh['ORAINI'] = pd.to_datetime(qh['ORAINI'], format='%Y%m%d%H%M')
    qh.set_index('ORAINI', inplace=True)
    # Nord
    qh_nord = qh[qh['MACROZONA'] == 'NORD']

    # Resample to hourly data and take the sum of 'SBIL_MWH' for each hour
    h_nord = qh_nord.resample('H').sum()

    # Clean the 'MACROZONA' column to retain only 'NORD'
    h_nord['MACROZONA'] = h_nord['MACROZONA'].str[:4]

    h_nord = h_nord.drop(columns = ['MACROZONA'])

    lags = [-1, -2, -3, -24]

    # Create a new DataFrame with SBIL_MWH and its lagged values
    df_sbil_lagged = pd.DataFrame(h_nord['SBIL_MWH'])

    # Add lagged columns
    for lag in lags:
        df_sbil_lagged[f'SBIL_MWH_lag{abs(lag)}'] = df_sbil_lagged['SBIL_MWH'].shift(-lag)

    df_sbil_lagged = df_sbil_lagged.drop('SBIL_MWH', axis=1)
    df_sbil_lagged = df_sbil_lagged.resample('H').sum()

    # Add the source directory to the system path
    sys.path.append(os.path.abspath('../../src'))
    from open_data import fetch_db_table_sqlserver16_
    import utils
    import importlib
    # Reload the module
    importlib.reload(utils)

    # SQL query to fetch data
    sql_query = """
    SELECT * FROM POWER_UNBALANCE
    """
    power_curve = fetch_db_table_sqlserver16(sql=sql_query, verbose=False)
    power_curve = power_curve.sort_values(by='TIMESTAMP')

    power_curve = power_curve.pivot(index='TIMESTAMP', columns='SOURCE_ZONE', values='UNBALANCE_kW')

    # Rename columns for clarity (optional, if needed)
    power_curve.columns = [f"UNBALANCE_{col}" for col in power_curve.columns]

    # Display the resulting DataFrame
    power_curve = power_curve.resample('H').sum()

    power_curve = power_curve[["UNBALANCE_IDRO-NON-PROGRAMMABILE_MACRONORD", "UNBALANCE_IDRO-PROGRAMMABILE_NORD", "UNBALANCE_SOLARE_NORD"]]

    power_curve = power_curve.rename_axis("ORAINI")


    import sys
    import os
    import pandas as pd

    # Add the source directory to the system path
    sys.path.append(os.path.abspath('../../src'))
    from open_data import fetch_db_table_sqlserver16
    import utils
    import importlib

    # Reload the module
    importlib.reload(utils)


    from functools import reduce
    # List of all the DataFrames to be merged
    dataframes = [h_nord, df_sbil_lagged, power_curve] #mi1_volumes_nord  mgp_volumes_nord,
    # Use reduce to merge all DataFrames on 'ORAINI'
    df_nord_h_project = reduce(lambda left, right: pd.merge(left, right, on='ORAINI', how='outer'), dataframes)

    #df_nord_h = df_nord.drop(columns="MACROZONA")
    df_nord_h_project = df_nord_h_project[df_nord_h_project.index >= '2024-08-27']


    df_nord = df_nord_h_project

    # Check for duplicate timestamps in the index and remove duplicates
    df_nord = df_nord[~df_nord.index.duplicated(keep='first')]


    # Check for duplicate timestamps in the index and remove duplicates
    df_nord = df_nord[~df_nord.index.duplicated(keep='first')]

    # Define past, future, and present covariates
    past_covariates = []

    future_covariates = [] #df_nord[['MGP_NORD_PURCHASES', 'MGP_NORD_SALES']]

    present_covariates = df_nord[['SBIL_MWH_lag1', 'SBIL_MWH_lag2', 'SBIL_MWH_lag3', 'SBIL_MWH_lag24', 'UNBALANCE_IDRO-NON-PROGRAMMABILE_MACRONORD', 'UNBALANCE_IDRO-PROGRAMMABILE_NORD', 'UNBALANCE_SOLARE_NORD']]
    target = df_nord['SBIL_MWH']

    # Drop rows with NaN values resulting from the shift
    df_nord = df_nord.dropna()

    # Features (X) and Target (y)
    X = df_nord.drop(columns=['SBIL_MWH'])
    y = df_nord['SBIL_MWH']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # Add the Dropout-enhanced MLP Model
    model2 = Sequential()

    # Input layer + First hidden layer with L2 regularization and Dropout
    model2.add(Dense(8, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    model2.add(Dropout(0.2))  # Dropout with 20% probability

    # Second hidden layer with L2 regularization and Dropout
    model2.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
    model2.add(Dropout(0.2))  # Dropout with 20% probability

    # Output layer (single neuron for regression)
    model2.add(Dense(1))

    # Adam optimizer with initial learning rate 0.001
    optimizer = Adam(learning_rate=0.001)

    # Compile the model
    model2.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model
    history = model2.fit(X_train, y_train, epochs=150, batch_size=512, validation_split=0.2, callbacks=[early_stopping])



    # Evaluate the model on test data
    y_pred = model2.predict(X_test)

    # Inverse transform predictions and true values to get the original scale
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Number of future steps to predict
    n_future_steps = 4  # Predict 4 steps ahead

    # Get the last timestamp from the dataset
    last_timestamp = df_nord.index[-1]

    # Get the most recent data from the test set for the starting point
    last_input = X_test[-1].reshape(1, -1)  # Last row from test features

    # Placeholder for predictions
    predicted_values = []

    # Monte Carlo Dropout
    n_simulations = 100  # Perform 100 stochastic forward passes
    mc_predictions = []  # Placeholder for stochastic predictions

    current_input = last_input
    for step in range(n_future_steps):
        # Collect multiple stochastic predictions for MC Dropout
        stochastic_results = []
        for _ in range(n_simulations):
            stochastic_pred = model2(current_input, training=True)  # Enable dropout during prediction
            stochastic_results.append(stochastic_pred.numpy().flatten()[0])
        stochastic_results = np.array(stochastic_results)

        # Calculate mean and standard deviation
        mean_prediction = stochastic_results.mean()
        std_prediction = stochastic_results.std()

        # Save the results
        predicted_values.append((mean_prediction, std_prediction))
        mc_predictions.append(stochastic_results)

        # Update the input for the next prediction
        current_input = np.roll(current_input, shift=-1, axis=1)  # Shift all features to the left
        current_input[0, -1] = mean_prediction  # Replace the last feature with the predicted value

    # Extract means and standard deviations
    predicted_means = [mean for mean, _ in predicted_values]
    predicted_stds = [std for _, std in predicted_values]

    # Convert to the original scale
    predicted_means_orig = scaler_y.inverse_transform(np.array(predicted_means).reshape(-1, 1)).flatten()
    predicted_stds_orig = scaler_y.inverse_transform(np.array(predicted_stds).reshape(-1, 1)).flatten()

    # Calculate 95% confidence intervals
    z_score = 1.96
    ci_upper = predicted_means_orig + z_score * predicted_stds_orig
    ci_lower = predicted_means_orig - z_score * predicted_stds_orig

    # Add 1-hour intervals for the future steps
    future_dates = [last_timestamp + pd.Timedelta(hours=i + 1) for i in range(n_future_steps)]

    # Create a DataFrame to store the predicted future values
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_SBIL_MWH': predicted_means_orig,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })

    # Debug: Confirm the DataFrame structure
    print(f"Future DataFrame:\n{future_df}")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Get the last `n` actual observations (from the test set) for plotting
    n_last_obs = 20  # For example, plot the last 20 observed values
    last_observed_dates = df_nord.index[-n_last_obs:]  # Get the dates for the last observations
    last_observed_values = scaler_y.inverse_transform(y_test[-n_last_obs:].reshape(-1, 1)).flatten()

    last_observed_df = pd.DataFrame({
        'Date': last_observed_dates,
        'Actual_SBIL_MWH': last_observed_values
    })

    # Plot observed values
    plt.step(last_observed_df['Date'], last_observed_df['Actual_SBIL_MWH'], label='Observed SBIL_MWH', color='blue')

    # Plot predicted values
    plt.plot(future_df['Date'], future_df['Predicted_SBIL_MWH'], label='Predicted Future SBIL_MWH', color='red', linestyle='--', marker='o', markersize=10)

    # Plot confidence intervals
    plt.fill_between(
        future_df['Date'], future_df['CI_Lower'], future_df['CI_Upper'], color='red', alpha=0.2, label='95% Confidence Interval'
    )

    # Add labels and title
    plt.title('Observed and Predicted Future SBIL_MWH Values with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('SBIL_MWH')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.show()

    import os
    import pandas as pd
    from datetime import datetime

    # File path for storing predictions
    file_path = r'C:\imbalance_forecast\data\forecast_segno_orario.csv'

    # Step 1: Check if the file exists and is not empty
    if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
        # Read the existing DataFrame from the CSV file
        df_existing = pd.read_csv(file_path)
    else:
        # If the file does not exist or is empty, create an empty DataFrame with the necessary columns
        df_existing = pd.DataFrame(columns=['FLOWDATE', 'FLOWHOUR', 'Predicted_SBIL_MWH_hplus1', 
                                            'Predicted_SBIL_MWH_hplus2', 'Predicted_SBIL_MWH_hplus3', 
                                            'Predicted_SBIL_MWH_hplus4', 'Run_Timestamp', 'MACRO_INDEX'])

    # Step 2: Ensure the columns exist, and if the file is empty, create them
    if df_existing.empty:
        df_existing = pd.DataFrame(columns=['FLOWDATE', 'FLOWHOUR', 'Predicted_SBIL_MWH_hplus1', 
                                            'Predicted_SBIL_MWH_hplus2', 'Predicted_SBIL_MWH_hplus3', 
                                            'Predicted_SBIL_MWH_hplus4', 'Run_Timestamp', 'MACRO_INDEX'])

    # Step 3: Assign predictions from `future_df` to respective columns
    # Assuming `future_df` has predictions for h+1 in row 0, h+2 in row 1, etc.
    predicted_hplus1 = round(future_df.loc[0, 'Predicted_SBIL_MWH'], 3)
    predicted_hplus2 = round(future_df.loc[1, 'Predicted_SBIL_MWH'], 3)
    predicted_hplus3 = round(future_df.loc[2, 'Predicted_SBIL_MWH'], 3)
    predicted_hplus4 = round(future_df.loc[3, 'Predicted_SBIL_MWH'], 3)

    # Use the timestamp of the h+1 prediction for the new row
    run_timestamp = datetime.now()
    future_date = pd.to_datetime(future_df.loc[0, 'Date'])  # Ensure future_date is a datetime object
    flow_date = int(future_date.strftime('%Y%m%d'))  # FLOWDATE in YYYYMMDD format
    flow_hour = future_date.hour  # FLOWHOUR extracted as the hour
    macro_index = 'MACRONORD'

    # Step 4: Check if the FLOWDATE, FLOWHOUR, and MACRO_INDEX already exist in `df_existing`
    existing_row = df_existing[(df_existing['FLOWDATE'] == flow_date) & 
                            (df_existing['FLOWHOUR'] == flow_hour) & 
                            (df_existing['MACRO_INDEX'] == macro_index)]

    if not existing_row.empty:
        # Update existing row
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus1'] = float(predicted_hplus1)
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus2'] = float(predicted_hplus2)
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus3'] = float(predicted_hplus3)
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus4'] = float(predicted_hplus4)
        df_existing.loc[existing_row.index, 'Run_Timestamp'] = run_timestamp
    else:
        # Append a new row with predictions
        new_row = {
            'FLOWDATE': flow_date,
            'FLOWHOUR': flow_hour,
            'Predicted_SBIL_MWH_hplus1': float(predicted_hplus1),
            'Predicted_SBIL_MWH_hplus2': float(predicted_hplus2),
            'Predicted_SBIL_MWH_hplus3': float(predicted_hplus3),
            'Predicted_SBIL_MWH_hplus4': float(predicted_hplus4),
            'Run_Timestamp': run_timestamp,
            'MACRO_INDEX': macro_index
        }
        df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)

    # Step 5: Ensure all `Predicted_SBIL_MWH_hplus*` columns are rounded and numeric
    for col in ['Predicted_SBIL_MWH_hplus1', 'Predicted_SBIL_MWH_hplus2', 'Predicted_SBIL_MWH_hplus3', 'Predicted_SBIL_MWH_hplus4']:
        df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce').round(3)

    # Step 6: Save the updated DataFrame back to the CSV file
    df_existing.to_csv(file_path, index=False)

    print("Data successfully appended or updated in the file.")

    
    
    ######################################################################
    
    # SQL query to fetch data
    sql_query = "SELECT * FROM TERNA_SUNSET_SEGNO_QH" 
    qh = fetch_db_table_sqlserver16(sql=sql_query, verbose=False)
    qh = qh.sort_values(by='ORAINI')
    # Filter the DataFrame for the relevant columns and rows
    qh = qh[['MACROZONA', 'ORAINI', 'SBIL_MWH']]
    # Convert ORAINI to datetime using the correct method
    qh['ORAINI'] = pd.to_datetime(qh['ORAINI'], format='%Y%m%d%H%M')
    qh.set_index('ORAINI', inplace=True)
    # sud
    qh_sud = qh[qh['MACROZONA'] == 'SUD']
    
    # Resample to hourly data and take the sum of 'SBIL_MWH' for each hour
    h_sud = qh_sud.resample('H').sum()

    # Clean the 'MACROZONA' column to retain only 'NORD'
    h_sud['MACROZONA'] = h_sud['MACROZONA'].str[:3]

    h_sud = h_sud.drop(columns = ['MACROZONA'])
    
    lags = [-1, -2, -24, -25]  

    # Create a new DataFrame with SBIL_MWH and its lagged values
    df_sbil_lagged = pd.DataFrame(h_sud['SBIL_MWH'])

    # Add lagged columns
    for lag in lags:
        df_sbil_lagged[f'SBIL_MWH_lag{abs(lag)}'] = df_sbil_lagged['SBIL_MWH'].shift(-lag)

    df_sbil_lagged = df_sbil_lagged.drop('SBIL_MWH', axis=1)
    df_sbil_lagged = df_sbil_lagged.resample('H').sum()
    

    # Add the source directory to the system path
    sys.path.append(os.path.abspath('../../src'))
    from open_data import fetch_db_table_sqlserver16
    # Reload the module
    importlib.reload(utils)

    # SQL query to fetch data
    sql_query = """
    SELECT * FROM POWER_UNBALANCE
    """
    power = fetch_db_table_sqlserver16(sql=sql_query, verbose=False)
    power = power.sort_values(by='TIMESTAMP')
    
    msud_power_curve = power.pivot(index='TIMESTAMP', columns='SOURCE_ZONE', values='UNBALANCE_kW')

    # Rename columns for clarity (optional, if needed)
    msud_power_curve.columns = [f"UNBALANCE_{col}" for col in msud_power_curve.columns]

    # Display the resulting DataFrame
    #msud_power_curve = msud_power_curve.resample('15min').sum()
    msud_power_curve = msud_power_curve.resample('H').sum()

    msud_power_curve = msud_power_curve[["UNBALANCE_EOLICO_CALA", "UNBALANCE_EOLICO_CSUD", "UNBALANCE_EOLICO_SICI", "UNBALANCE_EOLICO_SUD", "UNBALANCE_IDRO-NON-PROGRAMMABILE-MACROSUD", "UNBALANCE_SOLARE_CNOR", "UNBALANCE_SOLARE_CSUD", "UNBALANCE_SOLARE_SARD", "UNBALANCE_SOLARE_SICI", "UNBALANCE_SOLARE_SUD"]]

    msud_power_curve = msud_power_curve.rename_axis("ORAINI")
    
    # Summing Purchases from all MI1 zones to get MSUD_PURCHASED for MI1
    msud_eolico = (msud_power_curve['UNBALANCE_EOLICO_CALA'] + 
                          msud_power_curve['UNBALANCE_EOLICO_CSUD'] + 
                          msud_power_curve['UNBALANCE_EOLICO_SICI'] + 
                          msud_power_curve['UNBALANCE_EOLICO_SUD'] )

    # Summing Sales from all MI1 zones to get MSUD_SALES for MI1
    msud_solare = (msud_power_curve['UNBALANCE_SOLARE_CNOR'] + 
                      msud_power_curve['UNBALANCE_SOLARE_CSUD'] + 
                      msud_power_curve['UNBALANCE_SOLARE_SARD'] + 
                      msud_power_curve['UNBALANCE_SOLARE_SICI'] + 
                      msud_power_curve['UNBALANCE_SOLARE_SUD'])

    # Create DataFrame for MSUD_PURCHASED and MSUD_SALES for MI1
    msud_port_unbalance = pd.DataFrame({
        'UNBALANCE_EOLICO_MACROSUD': msud_eolico,
        'UNBALANCE_SOLAR_MACROSUD': msud_solare,
    })

    msud_port_unbalance['UNBALANCE_IDRO-NON-PROGRAMMABILE-MACROSUD'] = msud_power_curve['UNBALANCE_IDRO-NON-PROGRAMMABILE-MACROSUD']
    
    from functools import reduce
    # List of all the DataFrames to be merged
    dataframes = [h_sud, df_sbil_lagged, msud_port_unbalance]
    # Use reduce to merge all DataFrames on 'ORAINI'
    df_sud_h_port = reduce(lambda left, right: pd.merge(left, right, on='ORAINI', how='outer'), dataframes)

    #df_sud_h = df_nord.drop(columns="MACROZONA")
    df_sud_h_port = df_sud_h_port[df_sud_h_port.index >= '2024-08-27']
    
    df_sud = df_sud_h_port

    # Check for duplicate timestamps in the index and remove duplicates
    df_sud = df_sud[~df_sud.index.duplicated(keep='first')]


    # Check for duplicate timestamps in the index and remove duplicates
    df_sud = df_sud[~df_sud.index.duplicated(keep='first')]

    # Define past, future, and present covariates
    past_covariates = []

    future_covariates = [] #df_nord[['MGP_NORD_PURCHASES', 'MGP_NORD_SALES']]

    present_covariates = df_sud[['SBIL_MWH_lag1', 'SBIL_MWH_lag2', 'SBIL_MWH_lag24', 'SBIL_MWH_lag25', 'UNBALANCE_IDRO-NON-PROGRAMMABILE-MACROSUD', 'UNBALANCE_EOLICO_MACROSUD', 'UNBALANCE_SOLAR_MACROSUD']]
    target = df_sud['SBIL_MWH']

    # Drop rows with NaN values resulting from the shift
    df_sud = df_sud.dropna()

    # Features (X) and Target (y)
    X = df_sud.drop(columns=['SBIL_MWH'])
    y = df_sud['SBIL_MWH']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # Add the Dropout-enhanced MLP Model
    model2 = Sequential()

    # Input layer + First hidden layer with L2 regularization and Dropout
    model2.add(Dense(8, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    model2.add(Dropout(0.2))  # Dropout with 20% probability

    # Second hidden layer with L2 regularization and Dropout
    model2.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
    model2.add(Dropout(0.2))  # Dropout with 20% probability

    # Output layer (single neuron for regression)
    model2.add(Dense(1))

    # Adam optimizer with initial learning rate 0.001
    optimizer = Adam(learning_rate=0.001)

    # Compile the model
    model2.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model
    history = model2.fit(X_train, y_train, epochs=150, batch_size=512, validation_split=0.2, callbacks=[early_stopping])



    # Evaluate the model on test data
    y_pred = model2.predict(X_test)

    # Inverse transform predictions and true values to get the original scale
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Number of future steps to predict
    n_future_steps = 4  # Predict 4 steps ahead

    # Get the last timestamp from the dataset
    last_timestamp = df_sud.index[-1]

    # Get the most recent data from the test set for the starting point
    last_input = X_test[-1].reshape(1, -1)  # Last row from test features

    # Placeholder for predictions
    predicted_values = []

    # Monte Carlo Dropout
    n_simulations = 100  # Perform 100 stochastic forward passes
    mc_predictions = []  # Placeholder for stochastic predictions

    current_input = last_input
    for step in range(n_future_steps):
        # Collect multiple stochastic predictions for MC Dropout
        stochastic_results = []
        for _ in range(n_simulations):
            stochastic_pred = model2(current_input, training=True)  # Enable dropout during prediction
            stochastic_results.append(stochastic_pred.numpy().flatten()[0])
        stochastic_results = np.array(stochastic_results)

        # Calculate mean and standard deviation
        mean_prediction = stochastic_results.mean()
        std_prediction = stochastic_results.std()

        # Save the results
        predicted_values.append((mean_prediction, std_prediction))
        mc_predictions.append(stochastic_results)

        # Update the input for the next prediction
        current_input = np.roll(current_input, shift=-1, axis=1)  # Shift all features to the left
        current_input[0, -1] = mean_prediction  # Replace the last feature with the predicted value

    # Extract means and standard deviations
    predicted_means = [mean for mean, _ in predicted_values]
    predicted_stds = [std for _, std in predicted_values]

    # Convert to the original scale
    predicted_means_orig = scaler_y.inverse_transform(np.array(predicted_means).reshape(-1, 1)).flatten()
    predicted_stds_orig = scaler_y.inverse_transform(np.array(predicted_stds).reshape(-1, 1)).flatten()

    # Calculate 95% confidence intervals
    z_score = 1.96
    ci_upper = predicted_means_orig + z_score * predicted_stds_orig
    ci_lower = predicted_means_orig - z_score * predicted_stds_orig

    # Add 1-hour intervals for the future steps
    future_dates = [last_timestamp + pd.Timedelta(hours=i + 1) for i in range(n_future_steps)]

    # Create a DataFrame to store the predicted future values
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_SBIL_MWH': predicted_means_orig,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })

    # Debug: Confirm the DataFrame structure
    print(f"Future DataFrame:\n{future_df}")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Get the last `n` actual observations (from the test set) for plotting
    n_last_obs = 20  # For example, plot the last 20 observed values
    last_observed_dates = df_nord.index[-n_last_obs:]  # Get the dates for the last observations
    last_observed_values = scaler_y.inverse_transform(y_test[-n_last_obs:].reshape(-1, 1)).flatten()

    last_observed_df = pd.DataFrame({
        'Date': last_observed_dates,
        'Actual_SBIL_MWH': last_observed_values
    })

    # Plot observed values
    plt.step(last_observed_df['Date'], last_observed_df['Actual_SBIL_MWH'], label='Observed SBIL_MWH', color='blue')

    # Plot predicted values
    plt.plot(future_df['Date'], future_df['Predicted_SBIL_MWH'], label='Predicted Future SBIL_MWH', color='red', linestyle='--', marker='o', markersize=10)

    # Plot confidence intervals
    plt.fill_between(
        future_df['Date'], future_df['CI_Lower'], future_df['CI_Upper'], color='red', alpha=0.2, label='95% Confidence Interval'
    )

    # Add labels and title
    plt.title('Observed and Predicted Future SBIL_MWH Values with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('SBIL_MWH')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    
    
    import os
    import pandas as pd
    from datetime import datetime

    # File path for storing predictions
    file_path = r'C:\imbalance_forecast\data\forecast_segno_orario.csv'

    # Step 1: Check if the file exists and is not empty
    if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
        # Read the existing DataFrame from the CSV file
        df_existing = pd.read_csv(file_path)
    else:
        # If the file does not exist or is empty, create an empty DataFrame with the necessary columns
        df_existing = pd.DataFrame(columns=['FLOWDATE', 'FLOWHOUR', 'Predicted_SBIL_MWH_hplus1', 
                                            'Predicted_SBIL_MWH_hplus2', 'Predicted_SBIL_MWH_hplus3', 
                                            'Predicted_SBIL_MWH_hplus4', 'Run_Timestamp', 'MACRO_INDEX'])

    # Step 2: Ensure the columns exist, and if the file is empty, create them
    if df_existing.empty:
        df_existing = pd.DataFrame(columns=['FLOWDATE', 'FLOWHOUR', 'Predicted_SBIL_MWH_hplus1', 
                                            'Predicted_SBIL_MWH_hplus2', 'Predicted_SBIL_MWH_hplus3', 
                                            'Predicted_SBIL_MWH_hplus4', 'Run_Timestamp', 'MACRO_INDEX'])

    # Step 3: Assign predictions from `future_df` to respective columns
    # Assuming `future_df` has predictions for h+1 in row 0, h+2 in row 1, etc.
    predicted_hplus1 = round(future_df.loc[0, 'Predicted_SBIL_MWH'], 3)
    predicted_hplus2 = round(future_df.loc[1, 'Predicted_SBIL_MWH'], 3)
    predicted_hplus3 = round(future_df.loc[2, 'Predicted_SBIL_MWH'], 3)
    predicted_hplus4 = round(future_df.loc[3, 'Predicted_SBIL_MWH'], 3)

    # Use the timestamp of the h+1 prediction for the new row
    run_timestamp = datetime.now()
    future_date = pd.to_datetime(future_df.loc[0, 'Date'])  # Ensure future_date is a datetime object
    flow_date = int(future_date.strftime('%Y%m%d'))  # FLOWDATE in YYYYMMDD format
    flow_hour = future_date.hour  # FLOWHOUR extracted as the hour
    macro_index = 'MACROSUD'

    # Step 4: Check if the FLOWDATE, FLOWHOUR, and MACRO_INDEX already exist in `df_existing`
    existing_row = df_existing[(df_existing['FLOWDATE'] == flow_date) & 
                            (df_existing['FLOWHOUR'] == flow_hour) & 
                            (df_existing['MACRO_INDEX'] == macro_index)]

    if not existing_row.empty:
        # Update existing row
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus1'] = float(predicted_hplus1)
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus2'] = float(predicted_hplus2)
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus3'] = float(predicted_hplus3)
        df_existing.loc[existing_row.index, 'Predicted_SBIL_MWH_hplus4'] = float(predicted_hplus4)
        df_existing.loc[existing_row.index, 'Run_Timestamp'] = run_timestamp
    else:
        # Append a new row with predictions
        new_row = {
            'FLOWDATE': flow_date,
            'FLOWHOUR': flow_hour,
            'Predicted_SBIL_MWH_hplus1': float(predicted_hplus1),
            'Predicted_SBIL_MWH_hplus2': float(predicted_hplus2),
            'Predicted_SBIL_MWH_hplus3': float(predicted_hplus3),
            'Predicted_SBIL_MWH_hplus4': float(predicted_hplus4),
            'Run_Timestamp': run_timestamp,
            'MACRO_INDEX': macro_index
        }
        df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)

    # Step 5: Ensure all `Predicted_SBIL_MWH_hplus*` columns are rounded and numeric
    for col in ['Predicted_SBIL_MWH_hplus1', 'Predicted_SBIL_MWH_hplus2', 'Predicted_SBIL_MWH_hplus3', 'Predicted_SBIL_MWH_hplus4']:
        df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce').round(3)

    # Step 6: Save the updated DataFrame back to the CSV file
    df_existing.to_csv(file_path, index=False)

    print("Data successfully appended or updated in the file.")
    time.sleep(3600)