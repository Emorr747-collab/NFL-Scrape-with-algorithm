import pandas as pd
import logging
from typing import List, Tuple, Dict
from pathlib import Path
import os
import re

def strip_special_chars(name: str) -> str:
    """
    Remove special characters (* and +) from player names while preserving the base name
    """
    return re.sub(r'[*+]+$', '', name.strip())

def find_exact_duplicates(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Find names that are the same player (exact match when removing * and + suffixes)
    """
    name_groups = {}
    seen_bases = set()
    
    # Sort names so base name (no special chars) comes first
    sorted_names = sorted(df['Player'].unique())
    
    for name in sorted_names:
        base_name = strip_special_chars(name)
        if base_name not in name_groups:
            name_groups[base_name] = []
        name_groups[base_name].append(name)
    
    # Only keep groups with multiple entries
    return {k: v for k, v in name_groups.items() if len(v) > 1}

def combine_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine statistics for duplicate player entries
    """
    logger = logging.getLogger(__name__)
    
    # Create a copy of the dataframe
    combined_df = df.copy()
    
    # Find duplicate names
    duplicate_groups = find_exact_duplicates(combined_df)
    
    if not duplicate_groups:
        logger.info("No duplicate players found")
        return combined_df
    
    # Log the duplicates found
    logger.info("Found the following duplicate player entries:")
    for base_name, variants in duplicate_groups.items():
        logger.info(f"\n{base_name}:")
        for variant in variants:
            player_data = combined_df[combined_df['Player'] == variant].iloc[0]
            logger.info(f"  {variant}: {player_data['First_Year']}-{player_data['Last_Year']} ({player_data['Years_Played']} years)")
    
    # Create mapping to base names (without special characters)
    name_mapping = {}
    for base_name, variants in duplicate_groups.items():
        for variant in variants:
            name_mapping[variant] = base_name
    
    # Replace names with their base versions
    combined_df['Player'] = combined_df['Player'].replace(name_mapping)
    
    # Prepare aggregation dictionary based on available columns
    numeric_columns = [
        'Games', 'Games_Started', 'Attempts', 'Rushing_Yards', 
        'Rushing_TDs', 'First_Downs', 'Fumbles', 'Years_Played'
    ]
    
    agg_dict = {}
    
    # Add sum aggregation for numeric columns that exist
    for col in numeric_columns:
        if col in combined_df.columns:
            agg_dict[col] = 'sum'
    
    # Special handling for other columns
    agg_dict.update({
        'First_Year': 'min',
        'Last_Year': 'max',
        'Last_Age': 'max'
    })
    
    # Handle Teams column if it exists
    if 'Teams' in combined_df.columns:
        agg_dict['Teams'] = lambda x: ', '.join(sorted(set(','.join(x).split(', '))))
    
    # Group and aggregate
    combined_df = combined_df.groupby('Player').agg(agg_dict).reset_index()
    
    # Recalculate derived statistics
    if all(col in combined_df.columns for col in ['Attempts', 'Rushing_Yards']):
        combined_df['Yards_Per_Attempt'] = (
            combined_df['Rushing_Yards'] / combined_df['Attempts']
        ).round(1)
    
    if all(col in combined_df.columns for col in ['Attempts', 'Rushing_TDs']):
        combined_df['TD_Rate'] = (
            combined_df['Rushing_TDs'] / combined_df['Attempts'] * 100
        ).round(1)
    
    if all(col in combined_df.columns for col in ['Attempts', 'First_Downs']):
        combined_df['First_Down_Rate'] = (
            combined_df['First_Downs'] / combined_df['Attempts'] * 100
        ).round(1)
    
    if all(col in combined_df.columns for col in ['Attempts', 'Fumbles']):
        combined_df['Fumble_Rate'] = (
            combined_df['Fumbles'] / combined_df['Attempts'] * 100
        ).round(1)
    
    # Update Years_Played based on First_Year and Last_Year
    if all(col in combined_df.columns for col in ['First_Year', 'Last_Year']):
        combined_df['Years_Played'] = combined_df['Last_Year'] - combined_df['First_Year'] + 1
    
    logger.info(f"\nCombined {sum(len(variants) - 1 for variants in duplicate_groups.values())} duplicate entries")
    
    return combined_df

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Find input file on desktop
        desktop = Path.home() / 'Desktop'
        input_file = desktop / 'rb_stats_career_2000_to_2023.csv'
        
        if not input_file.exists():
            raise FileNotFoundError(f"Could not find {input_file}")
        
        logger.info(f"Processing file: {input_file}")
        
        # Read the data
        df = pd.read_csv(input_file)
        logger.info(f"Read {len(df)} player records")
        
        # Combine duplicate player entries
        combined_df = combine_player_stats(df)
        
        # Create output filename
        output_file = desktop / 'rb_stats_career_2000_to_2023_combined.csv'
        
        # Save the combined dataset
        combined_df.to_csv(output_file, index=False)
        logger.info(f"\nSaved combined stats to: {output_file}")
        
        # Show summary
        logger.info(f"\nOriginal unique players: {len(df['Player'].unique())}")
        logger.info(f"Final unique players: {len(combined_df['Player'].unique())}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    main()