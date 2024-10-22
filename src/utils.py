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

def process_foreign_exchange(df, country):
    # Step 1: Filter the dataframe by country
    country_df = df[df['COUNTRY'].str.lower() == country.lower()]
    
    # Step 2: Drop the 'COUNTRY' column
    country_df = country_df.drop(columns=['COUNTRY'])
    
    # Step 3: Rename the 'PHYSICAL_FOREIGN_FLOW_MW' column to '<country>_MWQH'
    country_df = country_df.rename(columns={"PHYSICAL_FOREIGN_FLOW_MW": f"{country.upper()}_MWQH"})
    
    # Step 4: Resample to 15-minute intervals and forward fill missing values
    country_df = country_df.resample('15min').ffill()
    
    # Step 5: Adjust the '<country>_MWQH' column values by dividing by 4
    country_df[f'{country.upper()}_MWQH'] = country_df[f'{country.upper()}_MWQH'] / 4
    
    return country_df