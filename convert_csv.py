import csv
import io


def add_speaker_tag_to_csv(input_csv_content, speaker_tag="[S1]"):
    """
    Reads CSV data, adds a speaker tag to the text column, and returns the modified CSV data as a string.

    Args:
        input_csv_content (str): A string containing the CSV data.
        speaker_tag (str): The speaker tag to prepend to the text (default is "[S1]").

    Returns:
        str: A string containing the modified CSV data.
    """
    # Use io.StringIO to treat the string as a file
    input_file = io.StringIO(input_csv_content)
    output_file = io.StringIO() # To store the output

    # The delimiter is '|'
    csv_reader = csv.reader(input_file, delimiter='|')
    # The quoting should handle quotes within fields if they exist,
    # and the delimiter for writing should also be '|'
    csv_writer = csv.writer(output_file, delimiter='|', quoting=csv.QUOTE_MINIMAL)

    modified_rows = []
    for row in csv_reader:
        if len(row) >= 2:  # Ensure there are at least two columns (audio_path, text, ...)
            # The text is in the second column (index 1)
            original_text = row[1]
            # Prepend the speaker tag
            modified_text = f"{speaker_tag}{original_text}"
            row[1] = modified_text
            modified_rows.append(row)
        else:
            # Handle rows that don't have the expected number of columns,
            # e.g., log them or add them as is. Here, we'll add them as is.
            modified_rows.append(row)

    # Write all modified rows to the output_file (which is an io.StringIO object)
    for row in modified_rows:
        csv_writer.writerow(row)

    return output_file.getvalue()

# --- How to use it with your file ---
# 1. Specify your input and output file paths
input_file_path = '/home/ubuntu/work/dia-finetuning/metadata.csv' # Replace with your actual input file name
output_file_path = '/home/ubuntu/work/dia-finetuning/metadata_tag.csv' # Replace with your desired output file name

# 2. To process a file from your system:
try:
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        file_content = infile.read()
    
    print(f"--- Original content from {input_file_path} (first 5 lines if available) ---")
    for i, line in enumerate(file_content.splitlines()):
        if i < 5:
            print(line)
        else:
            break
    print("...\n")

    modified_csv_string = add_speaker_tag_to_csv(file_content)

    with open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write(modified_csv_string)
    
    print(f"--- Modified content written to {output_file_path} (first 5 lines if available) ---")
    for i, line in enumerate(modified_csv_string.splitlines()):
        if i < 5:
            print(line)
        else:
            break
    print("...")
    print(f"\nSuccessfully processed the file. Output saved to: {output_file_path}")

except FileNotFoundError:
    print(f"Error: The file '{input_file_path}' was not found.")
    print("Please make sure the file exists in the same directory as the script, or provide the full path.")
    print("\n--- Processing sample data instead ---")
    # Process the sample data string if the file is not found
    modified_sample_data = add_speaker_tag_to_csv(csv_data_string)
    print("\n--- Modified Sample Data (output) ---")
    print(modified_sample_data)

except Exception as e:
    print(f"An error occurred: {e}")
    print("\n--- Processing sample data instead due to error ---")
    # Process the sample data string if any other error occurs
    modified_sample_data = add_speaker_tag_to_csv(csv_data_string)
    print("\n--- Modified Sample Data (output) ---")
    print(modified_sample_data)

