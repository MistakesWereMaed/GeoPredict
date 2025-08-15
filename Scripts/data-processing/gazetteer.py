import spacy
import pandas as pd

from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm



PATH_DATA = '../Data/Processed/data.csv'

PATH_GEONAMES = '../Data/Unprocessed/GeoNames/US.txt'
PATH_GAZETTEER = '../Data/Processed/gazetteer.csv'
PATH_METADATA = '../Data/Processed/metadata.csv'



spacy.require_gpu()
nlp = spacy.load('en_core_web_sm')

def join_rows(df, col):
    workers = cpu_count()
    print(f"Using {workers} workers")
    df_reduced = pd.DataFrame(
        Parallel(n_jobs=workers, verbose=1)(
            delayed(join_rows_inner)(group) for _, group in df.groupby(col)
        )
    )
    return df_reduced

def join_rows_inner(group):
    combined_rows = []
    combined_row = {}
    for col in group.columns:
        if group[col].dtype == 'object':
            unique_values = group[col].dropna().unique()
            combined_row[col] = ', '.join(map(str, unique_values))
        elif pd.api.types.is_numeric_dtype(group[col]):
            if col in {'latitude', 'longitude'}:
                combined_row[col] = group[col].mean()
            elif col in 'index':
                combined_row[col] = group[col].iloc[0]
            else:
                combined_row[col] = group[col].sum()
        combined_rows.append(combined_row)
    return pd.DataFrame(combined_rows)

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in {'GPE', 'LOC', 'FAC'}]

def batch_process(df_text, batch_size=2000):
    results = []
    for start in tqdm(range(0, len(df_text), batch_size), desc="Processing Batches"):
        end = min(start + batch_size, len(df_text))
        batch = df_text.iloc[start:end].copy()
        
        batch_entities = batch['text'].map(extract_entities)
        batch.loc[:, 'entities'] = batch_entities
        results.append(batch)
    
    return pd.concat(results, axis=0)

def extract_geographic_entities(df_text):
    df_text['text'] = df_text['text'].fillna('').astype(str)

    return batch_process(df_text)

def get_place_metadata(location, df_gazetteer):
    matching_row = df_gazetteer[df_gazetteer['name'] == location]
    
    if matching_row.empty:
        alternates = df_gazetteer['alternatenames'].fillna('').str.split(',')
        matching_row = df_gazetteer[alternates.list.contains(location)]

    return matching_row.iloc[0] if not matching_row.empty else None

def get_all_metadata(df_text, df_gazetteer):
    rows = []
    for idx, loc_list in df_text['entities'].to_pandas().items():
        for location in loc_list:
            location = str(location).lower().strip()
            result = get_place_metadata(location, df_gazetteer)
            if result is not None:
                result['index'] = idx
                row = result.to_dict()
                row = {str(k): v for k, v in row.items()}
                flattened_row = {key: list(sub_dict.values())[0] for key, sub_dict in row.items()}
                rows.append(flattened_row)

    return pd.DataFrame(rows)

def create_gazetteer():
    print("Loading GeoNames file...")

    df_geo = pd.read_csv(PATH_GEONAMES, sep='\t', header=None, 
        names=[
            'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature_class', 'feature_code', 'country_code', 'cc2', 
            'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 'modification_date'
        ],
        low_memory=False
    )[['name', 'alternatenames', 'population', 'latitude', 'longitude']]

    for col in df_geo.select_dtypes(include='object').columns:
        df_geo[col] = df_geo[col].str.lower()

    print("Loading complete")
    print("Rows in GeoNames file: ", len(df_geo))

    print("-------------------------------------")
    print("Joining rows by 'name'...")

    df_geo_reduced = join_rows(df_geo,'name')
    df_geo_reduced.to_csv(PATH_GAZETTEER, index=False, encoding='utf-8')

    print("Join complete")
    print("Rows in reduced GeoNames file: ", len(df_geo_reduced))

def generate_metadata(df, path):
    print("Extracting geographic entities...")
    df_entities = extract_geographic_entities(df)

    print("Extraction complete")
    print("-------------------------------------")

    df_gazetteer = pd.read_csv(path)

    print("Generating metadata...")
    df_metadata = get_all_metadata(df_entities, df_gazetteer)

    if not df_metadata.empty:
        print("Generation complete")
        print("-------------------------------------")

        print("Concatenating metadata...")
        df_metadata = join_rows(df_metadata, 'index')
        df_metadata = df_metadata.to_pandas()

        print("Concatination complete")
        print("-------------------------------------")
        
        df_merged = df.merge(df_metadata, on="index", how="left")
        df_merged.drop(['text'], axis=1, inplace=True)

        metadata_rows = df_merged[~df_merged['name'].isna()]
        count = len(metadata_rows)
        
        print(f"Rows with metadata: {count} ({round(count / len(df_merged) * 100, 2)}%)\n")
        return df_merged
        
    else:
        print("No metadata generated")
        return None
    
def main():
    create_gazetteer()

    df = pd.read_csv(PATH_DATA, delimiter=',')
    temp_df = pd.DataFrame(df['index'])
    temp_df['text'] = df['text'] + ', ' + df['city'].astype(str) + ', ' + df['state'].astype(str)

    df_metadata = generate_metadata(temp_df, PATH_GAZETTEER)
    df_metadata.to_csv(PATH_METADATA, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()