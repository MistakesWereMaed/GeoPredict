import re
import geopandas as gpd

from shapely.geometry import Point
from datetime import datetime
from sklearn.model_selection import train_test_split

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

def split_data(df, path_train, path_test, path_dev):
    train, temp = train_test_split(df, test_size=0.1, random_state=42)
    dev, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(path_train, index=False, encoding='utf-8')
    test.to_csv(path_test, index=False, encoding='utf-8')
    dev.to_csv(path_dev, index=False, encoding='utf-8')

    print(f"Train length: {len(train)}")
    print(f"Test length: {len(test)}")
    print(f"Dev length: {len(dev)}")