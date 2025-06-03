#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def verify_audio_paths(csv_file, audio_dir=None):
    """
    Verify that all audio paths in the CSV file exist.
    
    Args:
        csv_file: Path to the CSV file
        audio_dir: Optional directory to look for audio files if not found at the exact path
    """
    print(f"Verifying audio paths in: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return False
    
    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print("Error: CSV file does not contain any data rows")
        return False
    
    # Skip header
    data_lines = lines[1:]
    total_files = len(data_lines)
    existing_files = 0
    missing_files = []
    
    for i, line in enumerate(data_lines, start=2):
        line = line.strip()
        if not line:
            continue
        
        # Extract audio path (first column before the first pipe)
        parts = line.split('|', 1)
        if len(parts) < 2:
            print(f"Error: Line {i} does not contain a pipe delimiter: {line}")
            continue
        
        audio_path = parts[0].strip('"')
        
        # Check if file exists directly
        file_exists = os.path.isfile(audio_path)
        
        # If not found directly and audio_dir provided, check there
        if not file_exists and audio_dir:
            base_name = os.path.basename(audio_path)
            alternative_path = os.path.join(audio_dir, base_name)
            if os.path.isfile(alternative_path):
                print(f"Found at alternative path: {alternative_path}")
                file_exists = True
        
        if file_exists:
            existing_files += 1
        else:
            missing_files.append(audio_path)
    
    # Print results
    if existing_files == total_files:
        print(f"✅ All {total_files} audio files were found!")
        return True
    else:
        print(f"❌ Found {existing_files} out of {total_files} audio files")
        print(f"Missing {len(missing_files)} files:")
        for i, file in enumerate(missing_files[:5], start=1):
            print(f"  {i}. {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False

def main():
    args = sys.argv[1:]
    
    if args and args[0] in ['-h', '--help']:
        print("Usage: verify_audio_paths.py [csv_file] [audio_dir]")
        print("  csv_file: Path to the CSV file (default: output.csv)")
        print("  audio_dir: Directory to look for audio files (default: nour)")
        return
    
    csv_file = args[0] if len(args) > 0 else "/home/ubuntu/work/dia-finetuning/output.csv"
    audio_dir = args[1] if len(args) > 1 else "/home/ubuntu/work/dia-finetuning/nour"
    
    verify_audio_paths(csv_file, audio_dir)

if __name__ == "__main__":
    main()