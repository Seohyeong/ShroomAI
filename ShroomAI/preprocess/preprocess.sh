#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <RAW_DATA_PATH> <PREPROCESS_DATA_PATH>"
    exit 1
fi

RAW_DATA_PATH="$1"
PROCESSED_DATA_PATH="$2"

cd $RAW_DATA_PATH

# get gbifID and identifier(url) 
awk '{print $1 "\t" $4}' multimedia.txt > multimedia_id_url.txt

# get useful column and filter phylum (using only Ascomycota and Basidiomycota)
awk -F '\t' 'BEGIN {OFS="\t"} {print $1, $18, $81, $87, $98, $99, $149, $158, $159, $160, $162, $166, $202}' occurrence.txt > filtered_occurrence.txt
grep -E 'Ascomycota|Basidiomycota' filtered_occurrence.txt > filtered_phylum_occurrence.txt

# this gets rid of header, paste header to the filtered txt
head -n 1 filtered_occurrence.txt > header.txt
cat header.txt filtered_phylum_occurrence.txt > temp.txt && mv temp.txt filtered_phylum_occurrence.txt
rm header.txt

# filter out the rows with empty value of species column
awk '$13 != ""' filtered_phylum_occurrence.txt > filtered_phylum_species_occurrence.txt

rm filtered_occurrence.txt
rm filtered_phylum_occurrence.txt

# move processed files to processed data path
mv multimedia_id_url.txt $PROCESSED_DATA_PATH
mv filtered_phylum_species_occurrence.txt $PROCESSED_DATA_PATH