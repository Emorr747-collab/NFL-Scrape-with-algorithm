import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_total_position_counts(soup, year: int) -> dict:
    """
    Get the total number of players drafted by position
    """
    try:
        table = soup.find('table', {'id': 'drafts'})
        if not table:
            return {}
            
        all_positions = {}
        for row in table.find_all('tr'):
            # Skip header rows
            if 'class' in row.attrs and ('thead' in row.attrs['class'] or 'divider' in row.attrs['class']):
                continue
                
            pos_cell = row.find('td', {'data-stat': 'pos'})
            if pos_cell and pos_cell.text.strip() in ['QB', 'RB', 'WR', 'TE']:
                pos = pos_cell.text.strip()
                all_positions[pos] = all_positions.get(pos, 0) + 1
                
        return all_positions
    except Exception as e:
        logger.error(f"Error getting position counts for {year}: {str(e)}")
        return {}

def get_college_initials(college_name: str) -> str:
    """
    Get initials from college name, handling special cases
    """
    if pd.isna(college_name) or college_name == '':
        return 'UNK'
        
    # Handle special cases
    special_cases = {
        'Louisiana State': 'LSU',
        'Southern California': 'USC',
        'Central Florida': 'UCF',
        'Miami (FL)': 'MIA',
        'Miami (OH)': 'MIO',
        'Texas Christian': 'TCU',
        'Mississippi': 'MISS',
        'Mississippi State': 'MSU',
        'Southern Methodist': 'SMU',
        'Brigham Young': 'BYU',
        'California': 'CAL',
        'Pittsburgh': 'PITT',
        'Pennsylvania State': 'PSU',
        'North Carolina State': 'NCST',
        'South Carolina State': 'SCST'
    }
    
    if college_name in special_cases:
        return special_cases[college_name]
    
    # For other colleges, take first letter of each word
    words = college_name.split()
    if len(words) == 1:
        # If single word, take first 3 letters
        return words[0][:3].upper()
    else:
        # Take first letter of each word
        return ''.join(word[0] for word in words).upper()

def handle_duplicate_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add college initials to player names that are duplicates
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Count occurrences of each player name
    name_counts = df['Player'].value_counts()
    duplicate_names = name_counts[name_counts > 1].index
    
    # Process each duplicate name
    for name in duplicate_names:
        # Get all rows with this player name
        mask = df['Player'] == name
        duplicates = df[mask]
        
        logger.info(f"\nFound duplicate name: {name}")
        for _, row in duplicates.iterrows():
            logger.info(f"  Year: {row['Year']}, College: {row['College']}")
        
        # Add college initials to these players' names
        for idx in duplicates.index:
            college = df.loc[idx, 'College']
            initials = get_college_initials(college)
            df.loc[idx, 'Player'] = f"{name} ({initials})"
            
        logger.info("Updated names:")
        for _, row in df[df['Player'].str.startswith(name)].iterrows():
            logger.info(f"  {row['Player']}")
    
    return df

def scrape_draft_year(year: int) -> pd.DataFrame:
    """
    Scrape draft data for offensive skill positions (QB, RB, WR, TE) for a specific year
    """
    url = f"https://www.pro-football-reference.com/years/{year}/draft.htm"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        time.sleep(3)  # Respect rate limits
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get total position counts first
        total_positions = get_total_position_counts(soup, year)
        
        table = soup.find('table', {'id': 'drafts'})
        if not table:
            logger.warning(f"No draft table found for year {year}")
            return pd.DataFrame()

        # Get all column headers
        headers = []
        header_row = table.find('thead').find_all('tr')[-1]
        for th in header_row.find_all(['th']):
            stat = th.get('data-stat')
            if stat:
                headers.append(stat)
        
        logger.info(f"Found columns for year {year}: {headers}")
        
        # Track positions as we scrape
        scraped_positions = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        
        rows = []
        for row in table.find_all('tr'):
            if 'class' in row.attrs and ('thead' in row.attrs['class'] or 'divider' in row.attrs['class']):
                continue
                
            pos_cell = row.find('td', {'data-stat': 'pos'})
            if not pos_cell:
                continue
                
            position = pos_cell.text.strip()
            if position in ['QB', 'RB', 'WR', 'TE']:
                scraped_positions[position] += 1
                row_data = {}
                
                for header in headers:
                    cell = row.find(['td', 'th'], {'data-stat': header})
                    if cell:
                        value = cell.text.strip()
                        
                        if header == 'player':
                            player_link = cell.find('a')
                            if player_link:
                                value = player_link.text.strip()
                                row_data['player_id'] = player_link.get('href', '').split('/')[-1].replace('.htm', '')
                        
                        if header == 'college_id':
                            college_link = cell.find('a')
                            if college_link:
                                row_data['college_name'] = college_link.text.strip()
                                row_data['college_id'] = college_link.get('href', '').split('/')[-1].replace('.htm', '')
                            else:
                                row_data['college_name'] = value
                                row_data['college_id'] = ''
                        
                        row_data[header] = value
                
                if row_data:
                    rows.append(row_data)
        
        # Log position verification
        logger.info(f"\nPosition verification for {year}:")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            scraped = scraped_positions[pos]
            total = total_positions.get(pos, 0)
            status = "✓" if scraped == total else "✗"
            logger.info(f"{pos:3} {scraped:2d}/{total:2d} {status}")
            
            if scraped != total:
                logger.warning(f"Missing {total - scraped} {pos}s in {year}")
        
        if not rows:
            logger.warning(f"No offensive skill position players found for year {year}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df['draft_year'] = year
        
        # Column mapping and cleanup
        desired_columns = {
            'draft_year': 'Year',
            'draft_round': 'Round',
            'draft_pick': 'Pick',
            'player': 'Player',
            'player_id': 'Player_ID',
            'pos': 'Position',
            'college_name': 'College',
            'college_id': 'College_ID',
            'all_pros_first_team': 'AP1',
            'pro_bowls': 'PB',
            'years_played': 'Seasons_Played'
        }
        
        existing_columns = [col for col in desired_columns.keys() if col in df.columns]
        df = df[existing_columns]
        df = df.rename(columns={col: desired_columns[col] for col in existing_columns})
        
        # Clean player names before handling duplicates
        df['Player'] = df['Player'].str.replace('*', '').str.replace('+', '').str.strip()
        
        return df
        
    except Exception as e:
        logger.error(f"Error scraping year {year}: {str(e)}")
        return pd.DataFrame()

def main():
    all_draft_stats = pd.DataFrame()
    start_year = 1984
    end_year = 2024
    
    # Track total positions across all years
    total_positions = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    scraped_positions = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    
    for year in range(start_year, end_year):
        logger.info(f"\nProcessing draft year {year}")
        df_year = scrape_draft_year(year)
        
        if not df_year.empty:
            # Update position counts
            year_counts = df_year['Position'].value_counts()
            for pos in ['QB', 'RB', 'WR', 'TE']:
                scraped_positions[pos] += year_counts.get(pos, 0)
            
            if all_draft_stats.empty:
                all_draft_stats = df_year
            else:
                all_draft_stats = pd.concat([all_draft_stats, df_year], ignore_index=True)
        time.sleep(2)
    
    if not all_draft_stats.empty:
        # Handle duplicate names before saving
        logger.info("\nChecking for duplicate player names...")
        all_draft_stats = handle_duplicate_names(all_draft_stats)
        
        # Final position verification
        logger.info("\nFinal position totals across all years:")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            count = scraped_positions[pos]
            logger.info(f"{pos:3} {count:4d} players")
        
        output_dir = Path.home() / 'Desktop'
        output_file = output_dir / f'draft_stats_{start_year}_to_{end_year-1}.csv'
        
        all_draft_stats.to_csv(output_file, index=False)
        
        logger.info(f"\nDraft data saved to {output_file}")
        logger.info(f"Total draft picks processed: {len(all_draft_stats)}")
        
        # Show examples of handled duplicates
        duplicates = all_draft_stats[all_draft_stats['Player'].str.contains(r'\([A-Z]+\)')]
        if not duplicates.empty:
            logger.info("\nHandled duplicate names:")
            for _, row in duplicates.iterrows():
                logger.info(f"  {row['Player']} - {row['Year']}")
    else:
        logger.error("No data was collected")

if __name__ == "__main__":
    main()