import utils

import spacy
import cudf
import pandas as pd

from tqdm import tqdm

spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')

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
    df_gazetteer = cudf.from_pandas(df_gazetteer)   
    df_text_gpu = cudf.from_pandas(df_text)

    rows = []
    for idx, loc_list in df_text_gpu['entities'].to_pandas().items():
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
        df_metadata = cudf.from_pandas(df_metadata)
        df_metadata = utils.join_rows(df_metadata, 'index', use_gpu=True)
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