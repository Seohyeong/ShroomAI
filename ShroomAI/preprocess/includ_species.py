import pandas as pd
from collections import Counter

multimedia_id_df = pd.read_csv('/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/multimedia_id_url.txt', sep='\t', dtype=str)
occurrence_df = pd.read_csv('/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/filtered_phylum_species_occurrence.txt', sep = '\t', dtype=str)

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

combined_df = combine_multimedia_occurrence(multimedia_id_df, occurrence_df)

# filter by basisOfRecord
combined_df = combined_df[combined_df['basisOfRecord'] == 'HUMAN_OBSERVATION']
combined_df = combined_df.drop('basisOfRecord', axis=1)

# filter by species
combined_df = combined_df.dropna(subset = ['species'])
species_counter = Counter(list(combined_df['species']))


species_keep_edible = []
with open('/Users/seohyeong/Projects/ShroomAI/ShroomAI/preprocess/edible.txt', 'r') as file:
    for line in file:
        stripped_line = line.strip()
        species_keep_edible.append(stripped_line)
with open('/Users/seohyeong/Projects/ShroomAI/ShroomAI/preprocess/psilocybin.txt', 'r') as file:
    for line in file:
        stripped_line = line.strip()
        species_keep_edible.append(stripped_line)
species_keep_edible = list(set(species_keep_edible))

species_keep_count = []

for item in species_counter:
    if species_counter[item] > 100 and item not in species_keep_edible:
        species_keep_count.append(item)

print("edible: {}, others: {}".format(len(species_keep_edible), len(species_keep_count)))
print("total: {}".format(len(list(set(species_keep_edible + species_keep_count)))))

current_df = pd.read_csv('/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/sampled_df.txt', sep='\t')
sampled_df_species = list(current_df['species'].unique())
species_keep = []
for species in list(set(species_keep_edible + species_keep_count)):
    if species not in sampled_df_species:
        species_keep.append(species)

print("already existing: {}".format(len(sampled_df_species)))
print("edible + others - already existing = {}".format(len(species_keep)))

filtered_df = combined_df[combined_df['species'].isin(list(set(species_keep)))] # 828862
filtered_df.to_csv('/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/more_sampled_df.txt', sep='\t', index=False) 

# 2567 extra species are added: count > 100 + edible + psilocybin