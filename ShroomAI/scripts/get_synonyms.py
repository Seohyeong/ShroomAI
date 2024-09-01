# occurrence.txt
# get columns related to name: 158, 202, 203, 204, 205, 169, 170, 171, 172, 167, 149, 

# awk -F '\t' 'BEGIN {OFS="\t"} {print $158, $202, $203, $204, $205, $169, $170, $171, $172, $167, $149}' occurrence.txt > occ_tmp.txt
# grep -E 'Ascomycota|Basidiomycota' occ_tmp.txt > occ_tmp_out.txt
# head -n 1 occ_tmp.txt > header.txt
# cat header.txt occ_tmp_out.txt > temp.txt && mv temp.txt occ_names.txt
# rm header.txt
# rm occ_tmp.txt
# rm occ_tmp_out.txt

import pandas as pd
df_path = '/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/raw/occ_names.txt'
name_df = pd.read_csv(df_path, sep='\t', dtype=str) # 40196025
# name_df = name_df.drop_duplicates() # 953745
name_df = name_df.dropna(subset = ['species'])
name_df = name_df.drop('phylum', axis=1)
# name_df.to_csv('/Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/raw/occ_names.txt', sep='\t', index=False)
print()