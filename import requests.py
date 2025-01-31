import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set the base URL for Pro Football Reference
base_url = "https://www.pro-football-reference.com/years/{}/passing.htm"  # This is for QB stats

# Function to scrape the stats for a given year
def scrape_qb_stats(year):
    url = base_url.format(year)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Locate the table with player stats
    table = soup.find('table', {'id': 'passing'})

    # Extract headers
    headers = [th.getText() for th in table.find_all('th')][1:31]  # Modify based on columns needed

    # Extract rows
    rows = table.find_all('tr')[2:]  # Start after the header row

    # Extract player data
    player_data = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 0:  # Ignore empty rows
            player_data.append([col.getText() for col in cols])

    # Create a DataFrame
    df = pd.DataFrame(player_data, columns=headers)
    
    # Adding the year to the DataFrame for reference
    df['Year'] = year

    return df

# Now, we can scrape for multiple years and positions

# Set the range of years you want to scrape
years = list(range(2000, 2023))

# Create an empty DataFrame to store all stats
all_qb_stats = pd.DataFrame()

# Loop over each year and scrape data
for year in years:
    df_year = scrape_qb_stats(year)
    all_qb_stats = pd.concat([all_qb_stats, df_year])

# Save the scraped data to a CSV file
all_qb_stats.to_csv('qb_stats_2000_to_2023.csv', index=False)

print("Scraping completed and data saved to qb_stats_2000_to_2023.csv")
