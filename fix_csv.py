import pandas as pd
import unicodedata
import re

# Load CSV (with '|' separator and no headers)
df = pd.read_csv('/home/ubuntu/work/dia-finetuning/metadata.csv', delimiter='|', names=['file', 'text', 'lang'])

# Normalize Unicode to NFC
#df['text'] = df['text'].apply(lambda x: unicodedata.normalize('NFC', str(x)))

# Fix "ال " splits (e.g., "ال حياة" → "الحياة")
def fix_al_prefix(text):
    return re.sub(r'ال\s+(?=[\u0621-\u064A])', 'ال', text)

# Apply fix
df['text'] = df['text'].apply(fix_al_prefix)

# Double-check how many still broken
still_broken = df[df['text'].str.contains(r'ال\s+[\u0621-\u064A]')]
print(f"Still broken rows after fix: {len(still_broken)}")

# Save fixed CSV
df.to_csv('/home/ubuntu/work/dia-finetuning/metadata_fixed.csv', sep='|', index=False, header=False)
