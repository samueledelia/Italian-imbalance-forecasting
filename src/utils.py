import pandas as pd


def filter_and_rename_energy_source(df, energy_source, new_column_name):
    """
    Filters the DataFrame by the specified energy source, drops the ENERGY_SOURCE column, 
    and renames the renewable generation column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing renewable energy data.
    - energy_source (str): The energy source to filter (e.g., 'Wind').
    - new_column_name (str): The new name for the renewable generation column.

    Returns:
    - pd.DataFrame: A new DataFrame with the filtered data and renamed column.
    """
    # Filter the DataFrame for the specified energy source and create a copy
    filtered_df = df[df['ENERGY_SOURCE'] == energy_source].copy()
    
    # Drop the ENERGY_SOURCE column
    filtered_df.drop(columns=['ENERGY_SOURCE'], inplace=True)
    
    # Rename the renewable generation column
    filtered_df.rename(columns={'RENEWABLE_GENERATION_GWh': new_column_name}, inplace=True)
    
    return filtered_df


