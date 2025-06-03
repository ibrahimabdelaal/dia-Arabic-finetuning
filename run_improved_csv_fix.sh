#!/bin/bash

# Make the script executable
chmod +x fix_csv_improved.py

echo "Made fix_csv_improved.py executable"
echo ""
echo "Running improved CSV fix script..."
./fix_csv_improved.py

echo ""
echo "To use the fixed CSV file with LocalDiaDataset, run your training script with:"
echo "python -m dia.finetune [...other args...] --csv_path /home/ubuntu/work/dia-finetuning/output_fixed.csv"