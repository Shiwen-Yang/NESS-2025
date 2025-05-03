import pandas as pd
import numpy as np
import seaborn as sns
import holidays
from uszipcode import SearchEngine

def impute_missing_values(df, ignore_columns=None):
    """
    Imputes missing values:
    - For float or numeric columns, fills with mean.
    - For categorical or other columns, fills with mode.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        ignore_columns (list or None): List of columns to skip. Default is None.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df = df.copy()
    if ignore_columns is None:
        ignore_columns = []
        
    for col in df.columns:
        if col in ignore_columns:
            continue
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
            else:
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)
    return df


def cleaning(df):
    colnames_to_int = ['marital_status', 'high_education_ind', 'address_change_ind',  'policy_report_filed_ind']
    df[colnames_to_int] = df[colnames_to_int].astype(int)
    
    df['witness_present_ind'] = ["NP" if x == 0 else
                                 "P" if x == 1 else
                                 "DK" for x in df['witness_present_ind']]
    
    colnames_to_str = ['witness_present_ind','zip_code']
    df[colnames_to_str] = df[colnames_to_str].astype(str)
    
    df['claim_date']=pd.to_datetime(df['claim_date'])
    df['zip_code'] = df['zip_code'].str.zfill(5)
    
    return(df)

def age_cap(df, age_cap, age_col='age_of_driver'):
    df.loc[df[age_col] > age_cap, age_col] = age_cap
    return df

def assign_age_group(df, age_col='age_of_driver', new_col='age_group'):
    """
    Adds a categorical age group column to the dataframe based on age_of_driver.
    Groups:
        - '18-19'
        - '20-38'
        - '39-49'
        - '50-81'
        - '82-'
    """
    bins = [17, 19, 38, 49, 81, float('inf')]
    labels = ['A', 'B', 'C', 'D', 'E']
    df[new_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=True).astype('str')
    return df


def extract_datetime_features(df, date_col='claim_date', include_holidays=True):
    """
    Extracts basic datetime features from a given datetime column.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        date_col (str): The name of the datetime column.

    Returns:
        pd.DataFrame: DataFrame with new datetime-derived columns.
    """
    df = df.copy()
    
    # Ensure column is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Extract common date features
    df[f'{date_col}.year'] = df[date_col].dt.year.astype('str')
    df[f'{date_col}.month'] = df[date_col].dt.month.astype('str')
    df[f'{date_col}.day'] = df[date_col].dt.day.astype('str')
    df[f'{date_col}.dayofweek'] = df[date_col].dt.dayofweek.astype('str')
    df[f'{date_col}.weekofyear'] = df[date_col].dt.isocalendar().week.astype('str')
    
    # Additional datetime features
    df[f'{date_col}.quarter'] = df[date_col].dt.quarter.astype('str')
    df[f'{date_col}.is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)

    if include_holidays:
        us_holidays = holidays.US()
        df[f'{date_col}.is_holiday'] = (df[date_col].isin(us_holidays)).astype(int)
    
    df = df.drop(columns = ['claim_date', 'claim_day_of_week'])

    return df



def process_zipcode_features(df, zip_col='zip_code', plot=False):
    """
    Looks up and processes ZIP code-level features for a given DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a ZIP code column.
        zip_col (str): Name of the ZIP code column in the DataFrame.
        plot (bool): If True, plots histogram of log population.

    Returns:
        pd.DataFrame: Processed ZIP code DataFrame with selected features.
    """
    search_zip = SearchEngine()
    unique_zip = df[zip_col].unique()

    zip_code_basic_features = ['zipcode', 'zipcode_type', 'state', 'population']
    zip_code_lite = []

    for zip_code in unique_zip:
        zip_info = search_zip.by_zipcode(zip_code)
        if zip_info is not None:
            zip_dict = zip_info.to_dict()
            zip_dict_lite = {key: zip_dict.get(key, np.nan) for key in zip_code_basic_features}
            zip_code_lite.append(zip_dict_lite)

    # Handle ZIP code 0 case explicitly
    zero_dict_lite = {key: np.nan for key in zip_code_basic_features}
    zero_dict_lite['zipcode'] = '00000'
    zero_dict_lite['zipcode_type'] = 'UNIQUE'
    zip_code_lite.append(zero_dict_lite)

    zip_code_df = pd.DataFrame(zip_code_lite)
    zip_code_df['zipcode'] = zip_code_df['zipcode'].astype(str).str.zfill(5)

    # Compute log population
    zip_code_df['log_population'] = np.log1p(zip_code_df['population'])

    # Bin log population
    zip_code_df['log_pop_bin'] = 0
    bins = [0, 5, 10, np.inf]
    labels = [1, 2, 3]
    non_null_mask = zip_code_df['log_population'].notnull()

    zip_code_df.loc[non_null_mask, 'log_pop_bin'] = pd.cut(
        zip_code_df.loc[non_null_mask, 'log_population'],
        bins=bins,
        labels=labels,
        right=False
    ).astype(int)

    # Optional histogram
    if plot:
        sns.histplot(zip_code_df['log_population'], bins=15)
        
    zip_features = zip_code_df.drop(columns=['population', 'log_population'])
    
    df = df.merge(zip_features, left_on='zip_code', right_on='zipcode', how='left')
    df = df.drop(columns= ['zip_code', 'zipcode'])
    # Clean up
    return df

def price_categories(df, col = 'vehicle_price', new_col_name = 'vehicle_price_categories'):
    df = df.copy()
    df[new_col_name] = ['under_15k' if x<=15000  else 
                        'btw_20_30k' if 20000<=x<30000 else
                        'btw_30_40k' if 30000<=x<40000 else
                        'btw_40_50k' if 40000<=x<50000 else
                        'above_50k' for x in df[col]]
    return(df)

def liab_prct_group(df, col = 'liab_prct', new_col_name = 'liab_prct_group'):
    bins = [0, 5, 47.5, 52.5, 95, np.inf]
    labels = [0, 1, 2, 3, 4]

    df[new_col_name] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return(df)

def zero_payout(df, col = 'claim_est_payout', new_col_name = 'zero_payout'):
    df[new_col_name] = [1 if x > 0 else 0 for x in df[col]]
    return(df)


def add_interaction_features(df):
    """
    Adds meaningful interaction features based on financial and vehicle attributes.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with necessary columns.
        
    Returns:
        pd.DataFrame: DataFrame with additional interaction features.
    """
    df = df.copy()
    
    # Avoid division by zero or NaN propagation
    epsilon = 1e-5  # Small value to prevent divide-by-zero errors

    # Claim to income ratio
    df['claim_to_income_ratio'] = df['claim_est_payout'] / (df['annual_income'] + epsilon)

    # Vehicle price to income
    df['vehicle_price_to_income'] = df['vehicle_price'] / (df['annual_income'] + epsilon)
    
    # Income to age
    df['income_to_age'] = df['annual_income'] / (df['age_of_driver'] + epsilon)

    # Claim amount per age
    df['claim_amt_per_age'] = df['claim_est_payout'] / (df['age_of_driver'] + epsilon)
    
    # Claim number per age
    df['claim_nmb_per_age'] = df['past_num_of_claims'] / (df['age_of_driver'] + epsilon)
    
    # Claim number per vage
    df['claim_num_per_vage'] = df['past_num_of_claims'] / (df['age_of_vehicle'] + epsilon)

    # Age to vehicle price ratio
    df['price_to_age_ratio'] =  (df['vehicle_price'] )/ (df['age_of_driver']+ epsilon)

    # Vehicle price per weight
    df['price_per_weight'] = df['vehicle_price'] / (df['vehicle_weight'] + epsilon)

    # Liability-weighted claim
    df['liab_weighted_claim'] = df['liab_prct'] * df['claim_est_payout']

    # Marital × Income (if marital status is encoded)
    df['married_income'] = df['marital_status'].astype(int) * df['annual_income']

    return df



def preprocessing(df, age_cap_value=82, exclude_vars='witness_present_ind', include_holidays=True):
    """
    Runs the full preprocessing pipeline on the input dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        age_cap_value (int): Cap value for age_of_driver.
        exclude_vars (list): List of columns to exclude from imputation (default: None).
        include_holidays (bool): Whether to include holiday-based datetime features.

    Returns:
        pd.DataFrame: Fully processed dataframe.
    """
    df_processed = df.copy()
    
    df_processed = impute_missing_values(df_processed, exclude_vars)
    df_processed = cleaning(df_processed)
    df_processed = age_cap(df_processed, age_cap_value)
    df_processed = assign_age_group(df_processed)
    df_processed = extract_datetime_features(df_processed, include_holidays=include_holidays)
    df_processed = process_zipcode_features(df_processed)
    df_processed = price_categories(df_processed)
    df_processed = zero_payout(df_processed)
    df_processed = add_interaction_features(df_processed)
    
    return df_processed
