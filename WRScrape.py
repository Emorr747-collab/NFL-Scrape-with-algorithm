import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_to_100(df: pd.DataFrame, column: str, reverse: bool = False) -> pd.Series:
    """
    Normalize a column to 0-100 scale
    """
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return pd.Series([100 if max_val > 0 else 0] * len(df))
    
    if reverse:
        return 100 * (1 - (df[column] - min_val) / (max_val - min_val))
    return 100 * (df[column] - min_val) / (max_val - min_val)

def calculate_prime_years(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Calculate a player's prime 3-year stretch based on yards per reception
    """
    player_col = 'name_display' if 'name_display' in df.columns else 'player'
    
    prime_stats = {}
    min_receptions = 40 if position == 'WR' else 25  # Lower threshold for TEs
    
    for player in df[player_col].unique():
        player_data = df[df[player_col] == player].sort_values('Year')
        if len(player_data) >= 3:
            qualified_seasons = player_data[player_data['rec'] >= min_receptions]
            if len(qualified_seasons) >= 3:
                best_rating = 0
                best_years = None
                for i in range(len(qualified_seasons) - 2):
                    three_year_data = qualified_seasons.iloc[i:i+3]
                    # Calculate composite rating based on multiple factors
                    yards_per_rec = three_year_data['rec_yds'].sum() / three_year_data['rec'].sum()
                    td_rate = three_year_data['rec_td'].sum() / three_year_data['rec'].sum()
                    catch_rate = three_year_data['rec'].sum() / three_year_data['targets'].sum() if 'targets' in three_year_data.columns else 0
                    
                    composite_rating = (yards_per_rec * 0.4 + td_rate * 100 * 0.4 + catch_rate * 100 * 0.2)
                    
                    if composite_rating > best_rating:
                        best_rating = composite_rating
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

def scrape_receiver_stats(year: int) -> pd.DataFrame:
    """
    Scrape receiving statistics for a given year
    """
    url = f"https://www.pro-football-reference.com/years/{year}/receiving.htm"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        time.sleep(3)  # Respect rate limits
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'receiving'})
        
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
        
        df = pd.DataFrame(rows, columns=headers)
        df['Year'] = year
        
        # Filter for WR and TE only
        if 'pos' in df.columns:
            df = df[df['pos'].isin(['WR', 'TE'])].copy()
            logger.info(f"Found {len(df[df['pos'] == 'WR'])} WRs and {len(df[df['pos'] == 'TE'])} TEs for year {year}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error scraping year {year}: {str(e)}")
        return pd.DataFrame()

def process_receiver_data(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Process and aggregate receiver statistics for WR or TE
    
    Parameters:
    df (pd.DataFrame): DataFrame containing receiver statistics
    position (str): Position to filter for ('WR' or 'TE')
    
    Returns:
    pd.DataFrame: Processed and aggregated statistics
    """
    logger.info(f"Processing {len(df)} {position} records")
    
    # Filter for specific position
    df = df[df['pos'] == position].copy()
    
    # Clean player names
    def clean_player_name(name: str) -> str:
        return name.replace('*', '').replace('+', '').strip()
    
    if 'name_display' in df.columns:
        df['name_display'] = df['name_display'].apply(clean_player_name)
    elif 'player' in df.columns:
        df['player'] = df['player'].apply(clean_player_name)
    
    # Handle 2020 COVID season
    if 2020 in df['Year'].unique():
        try:
            covid_year = df[df['Year'] == 2020].copy()
            covid_year['g'] = pd.to_numeric(covid_year['g'], errors='coerce').fillna(0).astype(int)
            games_in_season = int(covid_year['g'].max())
            
            if games_in_season > 0 and games_in_season < 16:
                logger.info(f"Adjusting 2020 COVID season stats from {games_in_season} games to 16 games")
                multiplier = 16 / games_in_season
                adjust_cols = ['targets', 'rec', 'rec_yds', 'rec_td', 'rec_first_down']
                for col in adjust_cols:
                    if col in df.columns:
                        mask = df['Year'] == 2020
                        df.loc[mask, col] = pd.to_numeric(df.loc[mask, col], errors='coerce').fillna(0) * multiplier
        except Exception as e:
            logger.error(f"Error processing 2020 season: {str(e)}")
    
    # Convert numeric columns
    numeric_columns = [
        'age', 'g', 'gs', 'targets', 'rec', 'rec_yds', 
        'rec_td', 'rec_first_down', 'rec_long', 'rec_yds_per_rec',
        'rec_per_g', 'rec_yds_per_g', 'catch_pct', 'fumbles'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Determine player column
    player_col = 'name_display' if 'name_display' in df.columns else 'player'
    
    # Calculate prime years
    prime_years_df = calculate_prime_years(df, position)
    
    # Set up aggregation dictionary
    agg_dict = {
        'Year': ['min', 'max', 'count'],
        'age': 'last'
    }
    
    # Add numeric columns to aggregation
    if 'g' in df.columns:
        agg_dict['g'] = 'sum'
    if 'gs' in df.columns:
        agg_dict['gs'] = 'sum'
    if 'targets' in df.columns:
        agg_dict['targets'] = 'sum'
    if 'rec' in df.columns:
        agg_dict['rec'] = 'sum'
    if 'rec_yds' in df.columns:
        agg_dict['rec_yds'] = 'sum'
    if 'rec_td' in df.columns:
        agg_dict['rec_td'] = 'sum'
    if 'rec_first_down' in df.columns:
        agg_dict['rec_first_down'] = 'sum'
    if 'fumbles' in df.columns:
        agg_dict['fumbles'] = 'sum'
    if 'team' in df.columns:
        agg_dict['team'] = lambda x: ', '.join(sorted(set(x)))
    
    # Group and aggregate data
    grouped_df = df.groupby(player_col).agg(agg_dict).reset_index()
    grouped_df.columns = [f"{col[0]}{'_' + col[1] if col[1] else ''}" for col in grouped_df.columns]
    
    # Update column mapping
    column_mapping = {
        player_col: 'Player',
        'Year_min': 'First_Year',
        'Year_max': 'Last_Year',
        'Year_count': 'Years_Played',
        'age_last': 'Last_Age'
    }
    
    # Add position-specific mappings
    if 'g_sum' in grouped_df.columns:
        column_mapping['g_sum'] = 'Games'
    if 'gs_sum' in grouped_df.columns:
        column_mapping['gs_sum'] = 'Games_Started'
    if 'targets_sum' in grouped_df.columns:
        column_mapping['targets_sum'] = 'Targets'
    if 'rec_sum' in grouped_df.columns:
        column_mapping['rec_sum'] = 'Receptions'
    if 'rec_yds_sum' in grouped_df.columns:
        column_mapping['rec_yds_sum'] = 'Receiving_Yards'
    if 'rec_td_sum' in grouped_df.columns:
        column_mapping['rec_td_sum'] = 'Receiving_TDs'
    if 'rec_first_down_sum' in grouped_df.columns:
        column_mapping['rec_first_down_sum'] = 'First_Downs'
    if 'fumbles_sum' in grouped_df.columns:
        column_mapping['fumbles_sum'] = 'Fumbles'
    if 'team_<lambda>' in grouped_df.columns:
        column_mapping['team_<lambda>'] = 'Teams'
    
    grouped_df = grouped_df.rename(columns=column_mapping)
    
    # Calculate career statistics
    if all(col in grouped_df.columns for col in ['Receptions', 'Receiving_Yards']):
        grouped_df['Yards_Per_Reception'] = (
            grouped_df['Receiving_Yards'] / grouped_df['Receptions']
        ).round(1)
    
    if all(col in grouped_df.columns for col in ['Receptions', 'Targets']):
        grouped_df['Catch_Rate'] = (
            grouped_df['Receptions'] / grouped_df['Targets'] * 100
        ).round(1)
    
    if all(col in grouped_df.columns for col in ['Receiving_TDs', 'Receptions']):
        grouped_df['TD_Rate'] = (
            grouped_df['Receiving_TDs'] / grouped_df['Receptions'] * 100
        ).round(1)
        
    # Add per-game calculations
    if all(col in grouped_df.columns for col in ['Receiving_Yards', 'Games']):
        grouped_df['Yards_Per_Game'] = (
            grouped_df['Receiving_Yards'] / grouped_df['Games']
        ).round(1)

    if all(col in grouped_df.columns for col in ['Receiving_TDs', 'Games']):
        grouped_df['TD_Per_Game'] = (
            grouped_df['Receiving_TDs'] / grouped_df['Games']
        ).round(2)

    if all(col in grouped_df.columns for col in ['First_Downs', 'Receptions']):
        grouped_df['First_Down_Rate'] = (
            grouped_df['First_Downs'] / grouped_df['Receptions'] * 100
        ).round(1)

    # Set position-specific weights
    if position == 'WR':
        weights = {
            # Volume Stats (40%)
            'Receptions': 0.10,
            'Receiving_Yards': 0.20,
            'Receiving_TDs': 0.10,
            
            # Efficiency Stats (45%)
            'Yards_Per_Reception': 0.15,
            'Yards_Per_Game': 0.15,
            'TD_Per_Game': 0.15,
            
            # Reliability Stats (15%)
            'Catch_Rate': 0.08,
            'First_Down_Rate': 0.07
        }
    else:  # TE weights
        weights = {
            'Receptions': 0.20,
            'Receiving_Yards': 0.20,
            'Receiving_TDs': 0.25,
            'First_Downs': 0.15,
            'Yards_Per_Reception': 0.10,
            'Catch_Rate': 0.10,
        }

    # Work with all players
    working_df = grouped_df.copy()

    # Add experience level indicator
    working_df['Experience_Level'] = pd.cut(
        working_df['Receptions'],
        bins=[0, 100, 300, 700, float('inf')],
        labels=['Minimal', 'Limited', 'Moderate', 'Extended']
    )

    # Calculate normalized metrics
    for metric in weights:
        if metric in working_df.columns:
            working_df[f'Normalized_{metric}'] = normalize_to_100(
                working_df, 
                metric, 
                reverse=False
            )

    # Update era factors
    def get_era_factor(year: int) -> float:
        if year < 2004:
            return 1.12  # Slightly reduced from 1.15
        elif year < 2012:
            return 1.05  # Slightly reduced from 1.07
        else:
            return 1.00

    working_df['Era_Factor'] = working_df['First_Year'].apply(get_era_factor)

    # Calculate final score with better error handling
    working_df['Player_Score'] = 0.0  # Initialize with zeros
    for metric, weight in weights.items():
        if f'Normalized_{metric}' in working_df.columns:
            working_df['Player_Score'] += (
                weight * working_df[f'Normalized_{metric}'].fillna(0)
            )
    
    working_df['Player_Score'] = (working_df['Player_Score'] * working_df['Era_Factor']).clip(0, 100).round(1)

    # Define tier labels
    tier_labels = ['Below Average', 'Average', 'Good', 'Great', 'Elite', 'Unranked']
    
    try:
        # Create bins for valid scores
        valid_scores = working_df['Player_Score'] > 0
        
        if valid_scores.any():
            # Initialize all tiers as Unranked first
            working_df['Tier'] = pd.Categorical(['Unranked'] * len(working_df), categories=tier_labels)
            
            # Calculate tiers only for valid scores
            valid_scores_df = working_df.loc[valid_scores, 'Player_Score']
            
            # Create tier assignments
            tier_assignments = pd.qcut(
                valid_scores_df,
                q=5,
                labels=['Below Average', 'Average', 'Good', 'Great', 'Elite']
            )
            
            # Convert tier assignments to the same categorical type
            tier_assignments = pd.Categorical(
                tier_assignments,
                categories=tier_labels,
                ordered=False
            )
            
            # Assign tiers only to valid scores
            working_df.loc[valid_scores, 'Tier'] = tier_assignments
            
        else:
            # If no valid scores, all are Unranked
            working_df['Tier'] = pd.Categorical(['Unranked'] * len(working_df), categories=tier_labels)
            
    except ValueError as e:
        logger.warning(f"Could not calculate tiers: {str(e)}")
        working_df['Tier'] = pd.Categorical(['Unranked'] * len(working_df), categories=tier_labels)

    # Calculate all-time rank
    working_df['All_Time_Rank'] = (
        working_df['Player_Score']
        .rank(ascending=False, method='min', na_option='bottom')
        .fillna(working_df['Player_Score'].count())
        .astype(int)
    )

    # Create final dataframe
    final_df = pd.merge(
        grouped_df,
        working_df[['Player', 'Player_Score', 'Tier', 'All_Time_Rank', 'Experience_Level']],
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
    start_year = 2000
    end_year = 2024
    
    # Create separate DataFrames for WR and TE
    all_wr_stats = pd.DataFrame()
    all_te_stats = pd.DataFrame()
    
    for year in range(start_year, end_year):
        logger.info(f"Processing year {year}")
        df_year = scrape_receiver_stats(year)
        
        if not df_year.empty:
            wr_stats = df_year[df_year['pos'] == 'WR'].copy()
            te_stats = df_year[df_year['pos'] == 'TE'].copy()
            
            if not all_wr_stats.empty:
                all_wr_stats = pd.concat([all_wr_stats, wr_stats], ignore_index=True)
            else:
                all_wr_stats = wr_stats
                
            if not all_te_stats.empty:
                all_te_stats = pd.concat([all_te_stats, te_stats], ignore_index=True)
            else:
                all_te_stats = te_stats
                
        time.sleep(2)
    
    # Process each position separately
    if not all_wr_stats.empty:
        processed_wr_stats = process_receiver_data(all_wr_stats, 'WR')
        output_dir = Path.home() / 'Desktop'
        wr_output_file = output_dir / f'wr_stats_career_{start_year}_to_{end_year-1}.csv'
        processed_wr_stats.to_csv(wr_output_file, index=False)
        logger.info(f"Saved WR career stats to {wr_output_file}")
        logger.info(f"Total WRs processed: {len(processed_wr_stats)}")
    
    if not all_te_stats.empty:
        processed_te_stats = process_receiver_data(all_te_stats, 'TE')
        output_dir = Path.home() / 'Desktop'
        te_output_file = output_dir / f'te_stats_career_{start_year}_to_{end_year-1}.csv'
        processed_te_stats.to_csv(te_output_file, index=False)
        logger.info(f"Saved TE career stats to {te_output_file}")
        logger.info(f"Total TEs processed: {len(processed_te_stats)}")

if __name__ == "__main__":
    main()