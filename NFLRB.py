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

def calculate_prime_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a player's prime 3-year stretch based on rushing yards per attempt
    """
    # Determine which column name to use for player
    player_col = 'name_display' if 'name_display' in df.columns else 'player'
    
    prime_stats = {}
    for player in df[player_col].unique():
        player_data = df[df[player_col] == player].sort_values('Year')
        if len(player_data) >= 3:
            qualified_seasons = player_data[player_data['rush_att'] >= 100]
            if len(qualified_seasons) >= 3:
                best_rating = 0
                best_years = None
                for i in range(len(qualified_seasons) - 2):
                    three_year_avg = qualified_seasons.iloc[i:i+3]['rush_yds_per_att'].mean()
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

def scrape_rb_stats(year: int) -> pd.DataFrame:
    """
    Scrape RB statistics for a given year
    """
    url = f"https://www.pro-football-reference.com/years/{year}/rushing.htm"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        time.sleep(3)  # Respect rate limits
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'rushing'})
        
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
            df = df[df['pos'] == 'RB'].copy()
            logger.info(f"Found {len(df)} RBs for year {year}")
        
        # Print column names for debugging
        logger.info(f"Columns in dataframe: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error scraping year {year}: {str(e)}")
        return pd.DataFrame()

def process_and_aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and aggregate RB statistics
    """
    logger.info(f"Processing {len(df)} total records")
    
    # Clean player names to handle asterisks and plus signs
    def clean_player_name(name: str) -> str:
        return name.replace('*', '').replace('+', '').strip()
    
    if 'name_display' in df.columns:
        df['name_display'] = df['name_display'].apply(clean_player_name)
    elif 'player' in df.columns:
        df['player'] = df['player'].apply(clean_player_name)
    
    df = df[df['pos'] == 'RB'].copy()
    
   # Handle 2020 COVID season
    if 2020 in df['Year'].unique():
        try:
            covid_year = df[df['Year'] == 2020].copy()
            # Convert 'g' column to numeric first, force to int
            covid_year['g'] = pd.to_numeric(covid_year['g'], errors='coerce').fillna(0).astype(int)
            games_in_season = int(covid_year['g'].max())  # Force to integer
            
            logger.info(f"Detected {games_in_season} games in 2020 season")
            
            if games_in_season > 0 and games_in_season < 16:  # If shortened season
                logger.info(f"Adjusting 2020 COVID season stats from {games_in_season} games to 16 games")
                multiplier = 16 / games_in_season
                adjust_cols = ['rush_att', 'rush_yds', 'rush_td', 'rush_first_down']
                for col in adjust_cols:
                    if col in df.columns:
                        mask = df['Year'] == 2020
                        # Convert column to numeric before multiplication
                        df.loc[mask, col] = pd.to_numeric(df.loc[mask, col], errors='coerce').fillna(0) * multiplier
        except Exception as e:
            logger.error(f"Error processing 2020 season: {str(e)}")
            # Continue processing without COVID adjustment if there's an error
    
    # Update numeric columns with correct names
    numeric_columns = [
        'age', 'g', 'gs',  
        'rush_att', 'rush_yds', 'rush_td', 
        'rush_first_down', 'rush_yds_per_att', 
        'rush_yds_per_g', 'fumbles'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Determine which column name to use for player
    player_col = 'name_display' if 'name_display' in df.columns else 'player'
    
    prime_years_df = calculate_prime_years(df)
    
    # Update aggregation dictionary based on actual column names
    agg_dict = {
        'Year': ['min', 'max', 'count'],
        'age': 'last'
    }
    
    # Add columns only if they exist in the DataFrame
    if 'g' in df.columns:
        agg_dict['g'] = 'sum'
    if 'gs' in df.columns:
        agg_dict['gs'] = 'sum'
    if 'rush_att' in df.columns:
        agg_dict['rush_att'] = 'sum'
    if 'rush_yds' in df.columns:
        agg_dict['rush_yds'] = 'sum'
    if 'rush_td' in df.columns:
        agg_dict['rush_td'] = 'sum'
    if 'rush_first_down' in df.columns:
        agg_dict['rush_first_down'] = 'sum'
    if 'fumbles' in df.columns:
        agg_dict['fumbles'] = 'sum'
    if 'team' in df.columns:
        agg_dict['team'] = lambda x: ', '.join(sorted(set(x)))
    
    # Print the aggregation dictionary for debugging
    logger.info(f"Aggregation dictionary: {agg_dict}")
    
    grouped_df = df.groupby(player_col).agg(agg_dict).reset_index()
    grouped_df.columns = [f"{col[0]}{'_' + col[1] if col[1] else ''}" for col in grouped_df.columns]

    # Update column mapping based on actual columns
    column_mapping = {
        player_col: 'Player',
        'Year_min': 'First_Year',
        'Year_max': 'Last_Year',
        'Year_count': 'Years_Played',
        'age_last': 'Last_Age'
    }
    
    if 'g_sum' in grouped_df.columns:
        column_mapping['g_sum'] = 'Games'
    if 'gs_sum' in grouped_df.columns:
        column_mapping['gs_sum'] = 'Games_Started'
    if 'rush_att_sum' in grouped_df.columns:
        column_mapping['rush_att_sum'] = 'Attempts'
    if 'rush_yds_sum' in grouped_df.columns:
        column_mapping['rush_yds_sum'] = 'Rushing_Yards'
    if 'rush_td_sum' in grouped_df.columns:
        column_mapping['rush_td_sum'] = 'Rushing_TDs'
    if 'rush_first_down_sum' in grouped_df.columns:
        column_mapping['rush_first_down_sum'] = 'First_Downs'
    if 'fumbles_sum' in grouped_df.columns:
        column_mapping['fumbles_sum'] = 'Fumbles'
    if 'team_<lambda>' in grouped_df.columns:
        column_mapping['team_<lambda>'] = 'Teams'
    
    grouped_df = grouped_df.rename(columns=column_mapping)
    
    # Calculate career statistics only if required columns exist
    if all(col in grouped_df.columns for col in ['Attempts', 'Rushing_Yards']):
        grouped_df['Yards_Per_Attempt'] = np.where(
            grouped_df['Attempts'] > 0,
            (grouped_df['Rushing_Yards'] / grouped_df['Attempts']).round(1),
            0
        )
    
    if all(col in grouped_df.columns for col in ['Attempts', 'Rushing_TDs']):
        grouped_df['TD_Rate'] = np.where(
            grouped_df['Attempts'] > 0,
            (grouped_df['Rushing_TDs'] / grouped_df['Attempts'] * 100).round(1),
            0
        )
    
    if all(col in grouped_df.columns for col in ['Attempts', 'First_Downs']):
        grouped_df['First_Down_Rate'] = np.where(
            grouped_df['Attempts'] > 0,
            (grouped_df['First_Downs'] / grouped_df['Attempts'] * 100).round(1),
            0
        )
    
    if all(col in grouped_df.columns for col in ['Attempts', 'Fumbles']):
        grouped_df['Fumble_Rate'] = np.where(
            grouped_df['Attempts'] > 0,
            (grouped_df['Fumbles'] / grouped_df['Attempts'] * 100).round(1),
            0
        )
    
    weights = {
        'Attempts': 0.05,
        'Rushing_Yards': 0.25,
        'Rushing_TDs': 0.30,
        'First_Downs': 0.15,
        'Yards_Per_Attempt': 0.20,
        'TD_Rate': 0.05,
        'First_Down_Rate': 0.15,
        'Fumble_Rate': -0.15
    }
    
    # Only include metrics in weights if they exist in the DataFrame
    weights = {k: v for k, v in weights.items() if k in grouped_df.columns}
    
    # Instead of filtering, work with all RBs
    working_df = grouped_df.copy()
    
    working_df['Experience_Level'] = pd.cut(
        working_df['Attempts'],
        bins=[0, 300, 750, 1500, float('inf')],
        labels=['Minimal', 'Limited', 'Moderate', 'Extended']
    )
    
    for metric in weights:
        working_df[f'Normalized_{metric}'] = normalize_to_100(working_df, metric, weights[metric] < 0)
    
    def get_era_factor(year: int) -> float:
        if year < 2004:
            return 1.15  # Pre-passing emphasis rules
        elif year < 2012:
            return 1.07  # Early running era
        else:
            return 1.00  # Modern era
    
    working_df['Era_Factor'] = working_df['First_Year'].apply(get_era_factor)
    
    working_df['RB_Score'] = sum(
        weight * working_df[f'Normalized_{metric}']
        for metric, weight in weights.items()
    ) * working_df['Era_Factor']
    
    working_df['RB_Score'] = working_df['RB_Score'].clip(0, 100).round(1)
    
    try:
        working_df['Tier'] = pd.qcut(
            working_df['RB_Score'],
            q=5,
            labels=['Below Average', 'Average', 'Good', 'Great', 'Elite'],
            duplicates='drop'
        )
    except ValueError:
        unique_scores = len(working_df['RB_Score'].unique())
        if unique_scores >= 2:
            try:
                working_df['Tier'] = pd.qcut(
                    working_df['RB_Score'],
                    q=min(unique_scores - 1, 5),
                    labels=['Below Average', 'Average', 'Good', 'Great', 'Elite'][:min(unique_scores - 1, 5)],
                    duplicates='drop'
                )
            except ValueError:
                working_df['Tier'] = 'Unranked'
        else:
            working_df['Tier'] = 'Unranked'
    
    working_df['All_Time_Rank'] = working_df['RB_Score'].rank(
        ascending=False,
        method='min'
    ).astype(int)
    
    # Merge with original grouped_df to keep all players
    final_df = pd.merge(
        grouped_df,
        working_df[['Player', 'RB_Score', 'Tier', 'All_Time_Rank', 'Experience_Level']],
        on='Player',
        how='left'
    )
    
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
    all_rb_stats = pd.DataFrame()
    start_year = 2000
    end_year = 2024
    
    for year in range(start_year, end_year):
        logger.info(f"Processing year {year}")
        df_year = scrape_rb_stats(year)
        if not df_year.empty:
            if all_rb_stats.empty:
                all_rb_stats = df_year
            else:
                all_rb_stats = pd.concat([all_rb_stats, df_year], ignore_index=True)
        time.sleep(2)
    
    if not all_rb_stats.empty:
        processed_stats = process_and_aggregate_data(all_rb_stats)
        
        from pathlib import Path
        output_dir = Path.home() / 'Desktop'
        raw_output_file = output_dir / f'rb_stats_raw_{start_year}_to_{end_year-1}.csv'
        processed_output_file = output_dir / f'rb_stats_career_{start_year}_to_{end_year-1}.csv'
        
        # Save the files
        all_rb_stats.to_csv(raw_output_file, index=False)
        processed_stats.to_csv(processed_output_file, index=False)
        
        logger.info(f"Raw RB data saved to {raw_output_file}")
        logger.info(f"Processed RB career stats saved to {processed_output_file}")
        logger.info(f"Total unique RBs: {len(processed_stats)}")
        
        logger.info("\nSample of RB career stats:")
        print(processed_stats.head())
    else:
        logger.error("No data was collected")

if __name__ == "__main__":
    main()