import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split



PATH_DATA = '../Data/Processed/data.csv'

PATH_TRAIN = '../Data/Training/train.csv'
PATH_TEST = '../Data/Training/test.csv'
PATH_DEV = '../Data/Training/dev.csv'


def main():
    CLUSTERS = 4
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=0)
    df = pd.read_csv(PATH_DATA)

    df['cluster'] = kmeans.fit_predict(df[['longitude', 'latitude']])
    df = df[['index', 'hour', 'weekday', 'cluster', 'latitude', 'longitude', 'text']]
    print(f"Number of samples: {len(df)}")

    train, temp = train_test_split(df, test_size=0.1, random_state=42)
    dev, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(PATH_TRAIN, index=False, encoding='utf-8')
    test.to_csv(PATH_TEST, index=False, encoding='utf-8')
    dev.to_csv(PATH_DEV, index=False, encoding='utf-8')

    print(f"Train length: {len(train)}")
    print(f"Test length: {len(test)}")
    print(f"Dev length: {len(dev)}")

if __name__ == "__main__":
    main()