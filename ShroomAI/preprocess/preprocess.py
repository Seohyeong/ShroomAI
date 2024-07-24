import argparse
from collections import Counter
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def is_valid_url(url):
    if str(url).startswith('http'):
        return True
    else:
        return False


def combine_multimedia_occurrence(multimedia_id_df, occurrence_df):
    # filter out invalid url
    multimedia_id_df = multimedia_id_df[multimedia_id_df['identifier'].apply(lambda x: is_valid_url(x))]
    multimedia_id_df = multimedia_id_df.drop_duplicates(subset=['gbifID'])
    combined_df = pd.merge(occurrence_df, multimedia_id_df, how='inner', on='gbifID')
    return combined_df


def preprocess(combined_df, images_per_class):
    # filter by basisOfRecord
    combined_df = combined_df[combined_df['basisOfRecord'] == 'HUMAN_OBSERVATION']
    combined_df = combined_df.drop('basisOfRecord', axis=1)

    # filter by species
    combined_df = combined_df.dropna(subset = ['species'])
    species_counter = Counter(list(combined_df['species']))
    species_keep = []

    # keep images_per_class per species
    for item in species_counter:
        if species_counter[item] > images_per_class:
            species_keep.append(item)

    filtered_df = combined_df[combined_df['species'].isin(species_keep)]
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description='GBIF Mushroom Dataset Preprocessing')

    parser.add_argument('--dataset_dir_path', type=str, 
                        default='/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset') 
    parser.add_argument('--images_per_class', type=int, default=1000)

    args = parser.parse_args()

    args.raw_dataset_dir_path = os.path.join(args.dataset_dir_path, 'raw')
    args.occurence_file_path = os.path.join(args.dataset_dir_path, 'filtered_phylum_species_occurrence.txt')
    args.multimedia_file_path = os.path.join(args.dataset_dir_path, 'multimedia_id_url.txt')
    args.filtered_df_path = os.path.join(args.dataset_dir_path, 'sampled_df.txt')

    if not os.path.exists(args.filtered_df_path):
        # preprocess using preprocess.sh first
        preprocess_cmd = f'./preprocess.sh {args.raw_dataset_dir_path} {args.dataset_dir_path}'
        os.system(preprocess_cmd)

        # ID means one occurrence, multiple photos could have been taken for a single ID
        # [multimedia_id_df]: ID duplication exists vs [occurrence_df]: no ID duplication
        multimedia_id_df = pd.read_csv(args.multimedia_file_path, sep='\t', dtype=str)
        occurrence_df = pd.read_csv(args.occurence_file_path, sep = '\t', dtype=str)

        combined_df = combine_multimedia_occurrence(multimedia_id_df, occurrence_df)
        filtered_df = preprocess(combined_df, args.images_per_class)

        filtered_df.to_csv(os.path.join(args.dataset_dir_path, 'sampled_df.txt'), sep='\t', index=False)
    else:
        filtered_df = pd.read_csv(args.filtered_df_path, sep='\t')


if __name__ == '__main__':
    main()