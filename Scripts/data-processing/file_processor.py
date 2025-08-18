import re
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point
from datetime import datetime

PATH_TEXT = '../Data/Unprocessed/full_text.txt'
PATH_STATE_CITY = '../Data/Unprocessed/state_city.txt'
PATH_BOUNDARIES = '../Data/Unprocessed/NaturalEarth/ne_110m_admin_0_countries.shp'

PATH_DATA = '../Data/Processed/data.csv'

def get_boundaries(path, name):
    boundaries = gpd.read_file(path)
    return boundaries[boundaries['NAME'] == name]

def filter_coordinates(df, boundaries):
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    filtered_gdf = gdf[gdf.geometry.within(boundaries.geometry.squeeze())]
    return filtered_gdf

def parse_timestamp(timestamp):
    if timestamp == 'nan': return 0, 0
    
    time_obj = datetime.fromisoformat(timestamp)
    hour = time_obj.hour
    day_of_week = time_obj.weekday()

    return str(hour), str(day_of_week)

def clean_tweet_text(text):
    # Remove mentions (@usernames)
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters (optional, can retain hashtags if useful)
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def clean_df(df):
    df['hour'], df['weekday'] = zip(*df['timestamp'].apply(parse_timestamp))
    df['text'] = df['text'].astype(str).apply(clean_tweet_text)
    df['index'] = df.index

    return df[['user_id', 'index', 'hour', 'weekday', 'city', 'state', 'latitude', 'longitude', 'text']]

def main():
    print("Loading Text file...")

    df = pd.read_csv(PATH_TEXT, delimiter='\t', na_values=["", "NA"], encoding='utf-8', encoding_errors='replace')
    state_city = pd.read_csv(PATH_STATE_CITY, delimiter='\t', na_values=["", "NA"], encoding='utf-8', encoding_errors='replace')

    print("Loading complete")
    print("Rows in Text file: ", len(df))
    print("-------------------------------------")
    print("Filtering non-US coordinates...")

    boundaries = get_boundaries(PATH_BOUNDARIES, 'United States of America')
    df = filter_coordinates(df, boundaries).drop('geometry', axis=1)
    state_city = filter_coordinates(state_city, boundaries).drop('geometry', axis=1)

    print("Filtering complete")
    print("Rows in filtered datafile: ", len(df))
    print("-------------------------------------")
    print("Cleaning data...")

    df = clean_df(pd.merge(df, state_city, on=['latitude', 'longitude'], how='inner'))
    df = df[['user_id', 'index', 'hour', 'weekday', 'city', 'state', 'latitude', 'longitude', 'text']]
    df.to_csv(PATH_DATA, index=False, encoding='utf-8')

    print("Cleaning complete")

if __name__ == "__main__":
    main()