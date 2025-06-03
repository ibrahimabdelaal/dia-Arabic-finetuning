#!/usr/bin/env python3
import csv
import os
from pathlib import Path

def fix_csv_file(input_file, output_file, audio_dir=None):
    """
    Fix the CSV file by properly handling the column separators and quotes.
    Also verify that audio files exist.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to write the fixed CSV file
        audio_dir: Optional directory to look for audio files if not found at the exact path
    """
    print(f"Fixing CSV file: {input_file}")
    
    # Header for the output file
    header = "audio_path|text|language\n"
    
    # Parse the CSV file line by line
    fixed_lines = [header]
    missing_files = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        # Process each data row
        for line_number, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Find the first pipe character (separates audio_path from text)
                first_pipe_index = line.find('|')
                if first_pipe_index == -1:
                    print(f"Error: Line {line_number} does not contain a pipe delimiter. Skipping line.")
                    continue
                
                # Extract audio path (remove surrounding quotes)
                audio_path = line[:first_pipe_index].strip('"')
                rest_of_line = line[first_pipe_index + 1:]
                
                # Find the last pipe character (separates text from language)
                last_pipe_index = rest_of_line.rfind('|')
                if last_pipe_index == -1:
                    print(f"Warning: Line {line_number} does not have a language column. Using default language.")
                    text = rest_of_line.strip()
                    language = "ar"  # Default language
                else:
                    text = rest_of_line[:last_pipe_index].strip()
                    language = rest_of_line[last_pipe_index + 1:].strip('"')
                
                # Verify audio file exists
                audio_file_exists = os.path.isfile(audio_path)
                
                # If file not found directly, check in the audio_dir
                if not audio_file_exists and audio_dir:
                    base_name = os.path.basename(audio_path)
                    alternative_path = os.path.join(audio_dir, base_name)
                    if os.path.isfile(alternative_path):
                        audio_path = alternative_path
                        audio_file_exists = True
                
                if not audio_file_exists:
                    missing_files.append(audio_path)
                    print(f"Warning: Audio file not found: {audio_path}")
                
                # Clean up the text by removing excess quotes
                text = text.replace('""', '"')
                
                # Create a fixed line
                fixed_line = f"{audio_path}|{text}|{language}"
                fixed_lines.append(fixed_line)
                
            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
                print(f"Line content: {line}")
    
    # Write fixed lines to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + "\n")
    
    print(f"Fixed CSV written to: {output_file}")
    print(f"Total lines processed: {len(fixed_lines) - 1}")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} audio files not found")
        for i, file in enumerate(missing_files[:5], start=1):
            print(f"  {i}. {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    else:
        print("All audio files verified successfully!")
    
    return output_file

def main():
    input_file = Path("/home/ubuntu/work/dia-finetuning/output.csv")
    output_file = Path("/home/ubuntu/work/dia-finetuning/output_fixed.csv")
    audio_dir = Path("/home/ubuntu/work/dia-finetuning/nour")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return
    
    if not audio_dir.exists():
        print(f"Warning: Audio directory {audio_dir} does not exist")
    
    fixed_file = fix_csv_file(input_file, output_file, audio_dir)
    print(f"CSV file has been fixed: {fixed_file}")
    print("Now you can use this fixed CSV file with your LocalDiaDataset")

if __name__ == "__main__":
    main()