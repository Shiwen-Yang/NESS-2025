import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_fraud_rate_by_category(df, category_col, target_col='fraud', figsize=(10, 6), title=None):
    """
    Plots fraud rate for each category of an integer-encoded categorical variable,
    with ±1.96 SE error bars.
    
    Args:
        df: DataFrame containing the data.
        category_col: Column name of the categorical variable (must be integer-encoded).
        target_col: Column name of the target (default 'fraud').
        figsize: Size of the figure.
        title: Optional custom title for the plot.
        
    Returns:
        None (shows a plot)
    """
    df = df.copy()
    
    # Calculate fraud rate and sample size for each category
    fraud_rates = df.groupby(category_col)[target_col].mean().sort_index()
    sample_counts = df[category_col].value_counts().sort_index()

    # Calculate standard errors
    standard_errors = np.sqrt(fraud_rates * (1 - fraud_rates) / sample_counts)
    error_bars = 1.96 * standard_errors  # Approximate 95% confidence interval

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        fraud_rates.index.astype(str),  # treat x as discrete strings
        fraud_rates.values,
        yerr=error_bars.values,
        capsize=5,
        color='skyblue',
        edgecolor='black'
    )

    # Labels and Title
    ax.set_xlabel(category_col, fontsize=16)
    ax.set_ylabel(f'{target_col.capitalize()} Rate', fontsize=16)
    ax.set_title(title if title else f'{target_col.capitalize()} Rate by {category_col}', fontsize=18)

    # Ticks
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()



def plot_smoothed_fraud_rate_by_date(df, date_col='claim_date', target_col='fraud', window=7, figsize=(12, 6), title=None):
    """
    Plots a moving average of fraud rate by date.
    
    Args:
        df: DataFrame containing the data.
        date_col: Column name of the date variable (must be in datetime format).
        target_col: Column name of the target (default 'fraud').
        window: Size of the moving average window (default 7 days).
        figsize: Size of the figure.
        title: Optional custom title for the plot.
        
    Returns:
        None (shows a plot)
    """
    df = df.copy()
    
    # Make sure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Calculate fraud rate per day
    fraud_rate_by_date = df.groupby(date_col)[target_col].mean().sort_index()

    # Apply moving average
    fraud_rate_smooth = fraud_rate_by_date.rolling(window=window, center=True, min_periods=1).mean()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fraud_rate_by_date.index, fraud_rate_by_date.values, color='lightgray', label='Daily Fraud Rate', alpha=0.6)
    ax.plot(fraud_rate_smooth.index, fraud_rate_smooth.values, color='blue', linewidth=2, label=f'{window}-Day Moving Average')
    
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel(f'{target_col.capitalize()} Rate', fontsize=14)
    ax.set_title(title if title else f'{target_col.capitalize()} Rate Over Time (Smoothed)', fontsize=16)
    
    # Rotate x-ticks for better visibility
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(axis='both', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

