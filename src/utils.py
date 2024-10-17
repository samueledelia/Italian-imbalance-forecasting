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
    filtered_df.rename(columns={'ENERGY_BALANCE_GWh': new_column_name}, inplace=True)
    
    return filtered_df


def filter_load_by_zona(df, zonas):
    """
    Filters the DataFrame to include only rows where the 'ZONA' is in the provided list of zonas.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing load data.
        zonas (list): A list of 'ZONA' values to filter by.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the specified 'ZONA' values.
    """
    # Filtering the DataFrame
    filtered_df = df[df['ZONA'].isin(zonas)]
    return filtered_df


def filter_scheduled_foreign_exchange(df, country):
    return df[df['COUNTRY'].str.lower() == country.lower()]