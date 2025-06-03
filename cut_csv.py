import csv

# Input and output file names
input_file = '/home/ubuntu/work/dia-finetuning/metadata.csv'
output_file = 'output.csv'
# Open the input and output files
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the first 10 rows to the output file
    for i, row in enumerate(reader):
        if i < 10:
            writer.writerow(row)
        else:
            break
