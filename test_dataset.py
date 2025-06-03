#!/usr/bin/env python3
import os
import torch
from pathlib import Path

# Import necessary modules
try:
    import dac
    from dia.config import DiaConfig
    from dia.dataset import LocalDiaDataset
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have the necessary packages installed.")
    exit(1)

def test_dataset():
    print("Testing dataset loading from fixed CSV")
    
    # First fix the CSV file
    try:
        from fix_csv import fix_csv_file
        input_csv = Path("/home/ubuntu/work/dia-finetuning/output.csv")
        fixed_csv = Path("/home/ubuntu/work/dia-finetuning/output_fixed.csv")
        fix_csv_file(input_csv, fixed_csv)
        csv_path = fixed_csv
    except Exception as e:
        print(f"Error fixing CSV file: {e}")
        print("Using original CSV file as fallback")
        csv_path = Path("/home/ubuntu/work/dia-finetuning/output.csv")
    
    # Path to your audio files (same directory as the ones in CSV)
    audio_root = Path("/home/ubuntu/work/dia-finetuning/nour")
    
    # Check if csv exists
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Check if audio directory exists
    if not audio_root.exists():
        print(f"Error: Audio directory not found at {audio_root}")
        return
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load DAC model
        print("Loading DAC model...")
        dac_model = dac.DAC.load(dac.utils.download())
        dac_model = dac_model.to(device)
        
        # Create minimal config
        print("Creating config...")
        config = DiaConfig.load("/home/ubuntu/work/dia-finetuning/dia/config.json")
        
        # Create dataset
        print(f"Creating dataset from {csv_path}...")
        dataset = LocalDiaDataset(csv_path, audio_root, config, dac_model)
        
        # Try to load each item
        print(f"Dataset contains {len(dataset)} items")
        print("Testing access to each item:")
        
        for i in range(len(dataset)):
            try:
                print(f"\nLoading item {i+1}/{len(dataset)}:")
                text, encoded, waveform = dataset[i]
                print(f"  - Text: {text[:50]}...")
                print(f"  - Encoded shape: {encoded.shape}")
                print(f"  - Waveform shape: {waveform.shape}")
                print("  - Successfully loaded!")
            except Exception as e:
                print(f"  - Error loading item {i}: {e}")
        
        print("\nDataset test completed!")
        
    except Exception as e:
        print(f"Error during dataset testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()