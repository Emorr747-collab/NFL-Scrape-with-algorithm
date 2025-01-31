import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import numpy as np
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add these functions before your process_and_aggregate_data function:

def normalize_to_100(df: pd.DataFrame, column: str, reverse: bool = False) -> pd.Series:
    """
    Normalize a column to 0-100 scale
    
    Args:
        df: DataFrame containing the data
        column: Name of column to normalize
        reverse: If True, reverse the scale (100 becomes worst)
    
    Returns:
        Normalized series on 0-100 scale
    """
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return pd.Series([100 if max_val > 0 else 0] * len(df))
    
    if reverse:
        return 100 * (1 - (df[column] - min_val) / (max_val - min_val))
    return 100 * (df[column] - min_val) / (max_val - min_val)

def split_record(record_str: Optional[str]) -> pd.Series:
    """
    Split a W-L-T record string into separate columns
    
    Args:
        record_str: String in format "W-L-T"
    
    Returns:
        Series with wins, losses, and ties
    """
    try:
        if pd.isna(record_str) or record_str == '':
            return pd.Series([0, 0, 0])
        parts = record_str.split('-')
        if len(parts) != 3:
            return pd.Series([0, 0, 0])
        return pd.Series([
            int(parts[0]) if parts[0].isdigit() else 0,
            int(parts[1]) if parts[1].isdigit() else 0,
            int(parts[2]) if parts[2].isdigit() else 0
        ])
    except Exception as e:
        logger.warning(f"Error splitting record {record_str}: {str(e)}")
        return pd.Series([0, 0, 0])

def get_era(year: int) -> str:
    """
    Determine which NFL passing era a year belongs to
    
    Args:
        year: NFL season year
    
    Returns:
        String indicating the era
    """
    if year < 1978: 
        return "Pre-Modern"
    elif year < 1995: 
        return "Run Heavy Era"
    elif year < 2004: 
        return "Early Modern"
    elif year < 2012: 
        return "Post-Ty Law Rule"
    else: 
        return "Modern Passing"

def calculate_prime_window(df: pd.DataFrame, window_size: int = 48) -> Dict:
    """
    Calculate a player's prime window based on a rolling average of passer rating
    Args:
        df: DataFrame with player stats by game
        window_size: Number of games to consider for prime window (default 48 ~ 3 seasons)
    Returns:
        Dictionary with prime years and stats
    """
    prime_windows = {}
    
    for player in df['name_display'].unique():
        player_data = df[df['name_display'] == player].sort_values('Year')
        if len(player_data) >= window_size:
            # Calculate rolling average of key stats
            rolling_stats = player_data.rolling(window_size, min_periods=1).mean()
            
            # Find peak window
            peak_index = rolling_stats['pass_rating'].idxmax()
            if pd.notna(peak_index):
                prime_start = player_data.loc[peak_index - window_size + 1 if peak_index >= window_size else 0, 'Year']
                prime_end = player_data.loc[peak_index, 'Year']
                
                # Get stats from prime window
                prime_window = player_data[(player_data['Year'] >= prime_start) & 
                                        (player_data['Year'] <= prime_end)]
                
                prime_windows[player] = {
                    'Prime_Start': prime_start,
                    'Prime_End': prime_end,
                    'Prime_Passer_Rating': prime_window['pass_rating'].mean(),
                    'Prime_TD_Rate': (prime_window['pass_td'].sum() / prime_window['pass_att'].sum() * 100),
                    'Prime_Int_Rate': (prime_window['pass_int'].sum() / prime_window['pass_att'].sum() * 100),
                    'Prime_YPA': prime_window['pass_yds'].sum() / prime_window['pass_att'].sum(),
                    'Prime_Win_Pct': prime_window['Wins'].sum() / (prime_window['Wins'].sum() + prime_window['Losses'].sum()) * 100
                }
    
    return prime_windows

def calculate_era_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate league averages for different eras
    """
    # Convert numeric columns
    numeric_columns = ['pass_rating', 'pass_yds', 'pass_td', 'pass_int', 'pass_cmp', 'pass_att']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add era
    df['Era'] = df['Year'].apply(get_era)
    
    # Calculate era averages
    era_stats = df.groupby('Era').agg({
        'pass_rating': 'mean',
        'pass_yds': 'mean',
        'pass_td': 'mean',
        'pass_int': 'mean',
        'pass_cmp': 'mean',
        'pass_att': 'mean'
    }).round(2)
    
    era_stats.columns = [f'Era_Avg_{col}' for col in era_stats.columns]
    
    return era_stats

def calculate_prime_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a player's prime 3-year stretch based on passer rating
    
    Args:
        df: DataFrame with player stats by year
    
    Returns:
        DataFrame with prime years and ratings
    """
    prime_stats = {}
    for player in df['name_display'].unique():
        player_data = df[df['name_display'] == player].sort_values('Year')
        if len(player_data) >= 3:
            qualified_seasons = player_data[player_data['pass_att'] >= 100]
            if len(qualified_seasons) >= 3:
                best_rating = 0
                best_years = None
                for i in range(len(qualified_seasons) - 2):
                    three_year_avg = qualified_seasons.iloc[i:i+3]['pass_rating'].mean()
                    if three_year_avg > best_rating:
                        best_rating = three_year_avg
                        best_years = (
                            qualified_seasons.iloc[i]['Year'],
                            qualified_seasons.iloc[i+2]['Year']
                        )
                if best_years:
                    prime_stats[player] = {
                        'Prime_Years': f"{best_years[0]}-{best_years[1]}",
                        'Prime_Rating': round(best_rating, 1)
                    }
    return pd.DataFrame.from_dict(prime_stats, orient='index')

def scrape_qb_stats(year: int) -> pd.DataFrame:
    """
    Scrape QB statistics for a given year
    
    Args:
        year: NFL season year
    
    Returns:
        DataFrame with QB statistics
    """
    url = f"https://www.pro-football-reference.com/years/{year}/passing.htm"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        time.sleep(3)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'passing'})
        
        if not table:
            logger.warning(f"No data table found for year {year}")
            return pd.DataFrame()

        header_row = table.find('thead').find_all('tr')[-1]
        headers = [th.get('data-stat') for th in header_row.find_all(['th']) if th.get('data-stat')]
        
        rows = []
        for row in table.select('tbody tr:not(.thead)'):
            if row.find('td', {'data-stat': True}):
                row_data = []
                for stat in headers:
                    cell = row.find(['td', 'th'], {'data-stat': stat})
                    value = cell.text.strip() if cell else ''
                    row_data.append(value)
                
                if any(row_data):
                    rows.append(row_data)
        
        if not rows:
            logger.warning(f"No rows collected for year {year}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=headers)
        df['Year'] = year
        
        if 'pos' in df.columns:
            df = df[df['pos'] == 'QB'].copy()
            logger.info(f"Found {len(df)} QBs for year {year}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error scraping year {year}: {str(e)}")
        return pd.DataFrame()

def process_and_aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and aggregate QB statistics
    """
    logger.info(f"Processing {len(df)} total records")
    
    df = df[df['pos'] == 'QB'].copy()
    
    record_cols = ['Wins', 'Losses', 'Ties']
    if 'qb_rec' in df.columns:
        df[record_cols] = df['qb_rec'].apply(split_record)
    else:
        for col in record_cols:
            df[col] = 0
    
    numeric_columns = [
        'age', 'games', 'games_started', 'pass_cmp', 'pass_att', 'pass_yds', 
        'pass_td', 'pass_int', 'pass_rating', 'pass_sacked', 
        'pass_sacked_yds', 'pass_yds_per_att', 'pass_adj_yds_per_att'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    prime_years_df = calculate_prime_years(df)
    
    agg_dict = {
        'Year': ['min', 'max', 'count'],
        'age': 'last',
        'games': 'sum',
        'games_started': 'sum',
        'pass_cmp': 'sum',
        'pass_att': 'sum',
        'pass_yds': 'sum',
        'pass_td': 'sum',
        'pass_int': 'sum',
        'pass_rating': lambda x: np.mean(x[x > 0]) if any(x > 0) else 0,
        'pass_sacked': 'sum',
        'pass_sacked_yds': 'sum',
        'Wins': 'sum',
        'Losses': 'sum',
        'Ties': 'sum'
    }
    
    if 'team_name_abbr' in df.columns:
        agg_dict['team_name_abbr'] = lambda x: ', '.join(sorted(set(x)))
    
    grouped_df = df.groupby('name_display').agg(agg_dict).reset_index()
    grouped_df.columns = [f"{col[0]}{'_' + col[1] if col[1] else ''}" for col in grouped_df.columns]
    
    column_mapping = {
        'name_display': 'Player',
        'Year_min': 'First_Year',
        'Year_max': 'Last_Year',
        'Year_count': 'Years_Played',
        'age_last': 'Last_Age',
        'games_sum': 'Games',
        'games_started_sum': 'Games_Started',
        'pass_cmp_sum': 'Completions',
        'pass_att_sum': 'Attempts',
        'pass_yds_sum': 'Passing_Yards',
        'pass_td_sum': 'Passing_TDs',
        'pass_int_sum': 'Interceptions',
        'pass_rating_<lambda>': 'Average_Passer_Rating',
        'pass_sacked_sum': 'Times_Sacked',
        'pass_sacked_yds_sum': 'Sack_Yards_Lost',
        'Wins_sum': 'Total_Wins',
        'Losses_sum': 'Total_Losses',
        'Ties_sum': 'Total_Ties'
    }
    
    if 'team_name_abbr_<lambda>' in grouped_df.columns:
        column_mapping['team_name_abbr_<lambda>'] = 'Teams'
    
    grouped_df = grouped_df.rename(columns=column_mapping)
    
    grouped_df['Completion_Percentage'] = np.where(
        grouped_df['Attempts'] > 0,
        (grouped_df['Completions'] / grouped_df['Attempts'] * 100).round(1),
        0
    )
    
    grouped_df['TD_Percentage'] = np.where(
        grouped_df['Attempts'] > 0,
        (grouped_df['Passing_TDs'] / grouped_df['Attempts'] * 100).round(1),
        0
    )
    
    grouped_df['Int_Percentage'] = np.where(
        grouped_df['Attempts'] > 0,
        (grouped_df['Interceptions'] / grouped_df['Attempts'] * 100).round(1),
        0
    )
    
    grouped_df['Yards_Per_Attempt'] = np.where(
        grouped_df['Attempts'] > 0,
        (grouped_df['Passing_Yards'] / grouped_df['Attempts']).round(1),
        0
    )
    
    grouped_df['Win_Percentage'] = np.where(
        (grouped_df['Total_Wins'] + grouped_df['Total_Losses']) > 0,
        (grouped_df['Total_Wins'] / (grouped_df['Total_Wins'] + grouped_df['Total_Losses']) * 100).round(1),
        0
    )

    metrics = {
        'Attempts': False,
        'Passing_Yards': False,
        'Passing_TDs': False,
        'Interceptions': True,
        'Average_Passer_Rating': False,
        'Completion_Percentage': False,
        'Yards_Per_Attempt': False,
        'Times_Sacked': True
    }
    
    # Remove the qualification filter and work with the full dataset
    working_df = grouped_df.copy()
    
    # Add an experience level indicator without filtering
    working_df['Experience_Level'] = pd.cut(
        working_df['Attempts'],
        bins=[0, 100, 500, 1500, float('inf')],
        labels=['Minimal', 'Limited', 'Moderate', 'Extended']
    )

    for metric, reverse in metrics.items():
        working_df[f'Normalized_{metric}'] = normalize_to_100(working_df, metric, reverse)
    
    def get_era_factor(year: int) -> float:
        if year < 2004:
            return 1.15
        elif year < 2012:
            return 1.07
        else:
            return 1.00
    
    working_df['Era_Factor'] = working_df['First_Year'].apply(get_era_factor)
    
    weights = {
        'Attempts': 0.05,
        'Passing_Yards': 0.25,
        'Passing_TDs': 0.30,
        'Interceptions': -0.15,
        'Average_Passer_Rating': 0.20,
        'Completion_Percentage': 0.05,
        'Yards_Per_Attempt': 0.15,
        'Times_Sacked': -0.05
    }
    
    working_df['QB_Score'] = sum(
        weight * working_df[f'Normalized_{metric}']
        for metric, weight in weights.items()
    ) * working_df['Era_Factor']
    
    working_df['QB_Score'] = working_df['QB_Score'].clip(0, 100).round(1)
    
    try:
        working_df['Tier'] = pd.qcut(
            working_df['QB_Score'],
            q=5,
            labels=['Below Average', 'Average', 'Good', 'Great', 'Elite'],
            duplicates='drop'
        )
    except ValueError:
        unique_scores = len(working_df['QB_Score'].unique())
        if unique_scores >= 2:
            try:
                working_df['Tier'] = pd.qcut(
                    working_df['QB_Score'],
                    q=min(unique_scores - 1, 5),
                    labels=['Below Average', 'Average', 'Good', 'Great', 'Elite'][:min(unique_scores - 1, 5)],
                    duplicates='drop'
                )
            except ValueError:
                working_df['Tier'] = 'Unranked'
        else:
            working_df['Tier'] = 'Unranked'
    
    working_df['All_Time_Rank'] = working_df['QB_Score'].rank(
        ascending=False,
        method='min'
    ).astype(int)
    
    # Single merge operation with working_df
    final_df = pd.merge(
        grouped_df,
        working_df[['Player', 'QB_Score', 'Tier', 'All_Time_Rank']],
        on='Player',
        how='left'
    )
    
    # Add prime years if available
    if not prime_years_df.empty:
        final_df = pd.merge(
            final_df,
            prime_years_df,
            left_on='Player',
            right_index=True,
            how='left'
        )
    
    return final_df
    
def main():
    all_qb_stats = pd.DataFrame()
    start_year = 2000
    end_year = 2024
    
    for year in range(start_year, end_year):
        logger.info(f"Processing year {year}")
        df_year = scrape_qb_stats(year)
        if not df_year.empty:
            if all_qb_stats.empty:
                all_qb_stats = df_year
            else:
                all_qb_stats = pd.concat([all_qb_stats, df_year], ignore_index=True)
        time.sleep(2)
    
    if not all_qb_stats.empty:
        # Process the data
        processed_stats = process_and_aggregate_data(all_qb_stats)
        
        # Calculate era averages
        era_averages = calculate_era_averages(all_qb_stats)

        # Add debugging logs here before era calculations
        logger.info("Available columns in processed_stats:")
        print(processed_stats.columns.tolist())
        logger.info("\nAvailable columns in era_averages:")
        print(era_averages.columns.tolist())

        logger.info("\nSample of processed_stats:")
        print(processed_stats.head())
        logger.info("\nSample of era_averages:")
        print(era_averages.head())
        
        # Add era data and calculations
        processed_stats['Era'] = processed_stats['First_Year'].apply(get_era)
        processed_stats = processed_stats.merge(era_averages, on='Era', how='left')
        
        # Calculate prime windows
        prime_windows = calculate_prime_window(all_qb_stats)
        prime_df = pd.DataFrame.from_dict(prime_windows, orient='index')
        
        # Add prime window data
        processed_stats = processed_stats.merge(
            prime_df,
            left_on='Player',
            right_index=True,
            how='left'
        )
        
        # Calculate relative to era averages
        stat_mapping = {
            'Average_Passer_Rating': 'pass_rating',
            'Passing_Yards': 'pass_yds',
            'Passing_TDs': 'pass_td'
        }

        for processed_stat, era_stat in stat_mapping.items():
            processed_stats[f'{processed_stat}_vs_era'] = (
            processed_stats[processed_stat] / processed_stats[f'Era_Avg_{era_stat}'] * 100 - 100
        ).round(1)
        
        # Save files
        from pathlib import Path
        output_dir = Path.home() / 'Desktop'
        raw_output_file = output_dir / f'qb_stats_raw_{start_year}_to_{end_year-1}.csv'
        processed_output_file = output_dir / f'qb_stats_career_{start_year}_to_{end_year-1}.csv'
        
        all_qb_stats.to_csv(raw_output_file, index=False)
        processed_stats.to_csv(processed_output_file, index=False)
        
        logger.info(f"Raw QB data saved to {raw_output_file}")
        logger.info(f"Processed QB career stats saved to {processed_output_file}")
        logger.info(f"Total unique QBs: {len(processed_stats)}")
        
        logger.info("\nSample of QB career stats:")
        print(processed_stats.head())
    else:
        logger.error("No data was collected")

if __name__ == "__main__":
    main()