#!/bin/bash

# Navigate to the directory containing the images
cd /Users/seohyeong/Projects/ShroomAI/ShroomAI/dataset/images/

# Loop through each file in the directory
for file in *; do
    # Check if the file has an extension
    if [[ "$file" == *.* ]]; then
        # Get the current extension
        current_ext="${file##*.}"
        # Check if the current extension is not already jpg
        if [[ "$current_ext" != "jpg" ]]; then
            # Rename the file to have .jpg extension
            mv "$file" "${file%.*}.jpg"
        fi
    fi
done