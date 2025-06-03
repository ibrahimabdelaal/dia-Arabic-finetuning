import zipfile
import os

# Set the directory containing your zip files
zip_dir = "/home/ubuntu/work/dia-finetuning/"  # Change this if your zip files are elsewhere

# Loop through files in the directoryimport zipfile
import os

# Set the path to your individual ZIP file
zip_file_path = "/home/ubuntu/work/dia-finetuning/csvfiles.zip"  # Replace with your zip file name

# Get the file name without .zip extension
base_name = os.path.splitext(os.path.basename(zip_file_path))[0]

# Create the folder to extract into
extract_folder = os.path.join(os.path.dirname(zip_file_path), base_name)
os.makedirs(extract_folder, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"Extracted '{zip_file_path}' to '{extract_folder}/'")

for filename in os.listdir(zip_dir):
    if filename.endswith(".zip"):
        zip_path = os.path.join(zip_dir, filename)
        folder_name = filename[:-4]  # Remove .zip
        extract_path = os.path.join(zip_dir, folder_name)
        
        # Create the folder if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"Extracted '{filename}' to '{folder_name}/'")
