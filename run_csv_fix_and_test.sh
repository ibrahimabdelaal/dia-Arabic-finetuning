#!/bin/bash

# Make the scripts executable
chmod +x fix_csv.py test_dataset.py

echo "Made fix_csv.py and test_dataset.py executable"

# Run the CSV fix script
echo "Running CSV fix script..."
./fix_csv.py

echo ""
echo "Running dataset test script..."
./test_dataset.py