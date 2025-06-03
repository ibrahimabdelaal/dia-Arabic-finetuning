from pathlib import Path
import os
import csv
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

import dac
from .config import DiaConfig



import pandas as pd
from pathlib import Path
import os # Make sure os is imported
import torch # Make sure torch is imported
import torchaudio # Make sure torchaudio is imported
import csv # Import the csv module
# Make sure dac, DiaConfig are imported if this class is in a separate file
# import dac
# from .config import DiaConfig # Adjust if necessary

class LocalDiaDataset(Dataset): # Assuming Dataset is from torch.utils.data
    """Load from a local CSV (sep='|') + an audio folder."""
    def __init__(self, csv_path: Path, audio_root: Path, config: DiaConfig, dac_model: dac.DAC):
        try:
            # If your CSV file HAS a header row that you want to ignore,
            # and you want to define your own column names with the `names` parameter.
            self.df = pd.read_csv(csv_path, sep="|", engine="python",
                                  names=["audio_path", "text", "language"], # Your desired internal names
                                  skiprows=1, # Explicitly skip the first (header) row of the CSV
                                  quoting=csv.QUOTE_NONE) # Use csv.QUOTE_NONE
        except Exception as e:
            print(f"Warning: Error loading CSV with pandas: {e}")
            print("Falling back to manual CSV parsing (ensure it also skips header correctly if needed)")
            
            # Fallback to manual parsing (ensure this also correctly skips a header if present)
            rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Skip header if you are certain the file always has one
                # For robustness, you might want to check if the first line looks like a header
                try:
                    next(f) # Skip the header line
                    print("Manual parser: Skipped header row.")
                except StopIteration:
                    print("Manual parser: CSV was empty after trying to skip header.")
                    self.df = pd.DataFrame(rows) # empty dataframe
                    # ... rest of __init__ for empty df handling ...
                    return


                for line_number, line in enumerate(f, 1): # Start line number from 1 for data
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        rows.append({
                            "audio_path": parts[0].strip('"\''),
                            "text": parts[1], # Assuming text doesn't need stripping of quotes here
                            "language": parts[2].strip('"\'')
                        })
                    elif len(parts) == 2:
                        # Handle lines with only 2 parts, e.g., default language
                        print(f"Manual parser: Line {line_number+1} has only 2 parts, defaulting language to 'ar'. Line: '{line.strip()}'")
                        rows.append({
                            "audio_path": parts[0].strip('"\''),
                            "text": parts[1],
                            "language": "ar"  # Default language
                        })
                    else:
                        print(f"Manual parser: Skipping malformed line {line_number+1} (expected 2 or 3 parts): '{line.strip()}'")

            if not rows:
                 print("Warning: No data loaded after manual parsing attempts (or after skipping header). DataFrame will be empty.")
            self.df = pd.DataFrame(rows)
            
        if self.df.empty:
            print(f"Warning: DataFrame is empty after attempting to load CSV: {csv_path}")

        self.audio_root = Path(audio_root) # Ensure audio_root is a Path object
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        if idx >= len(self.df):
            raise IndexError("Index out of bounds")
        row = self.df.iloc[idx]
        
        lang = row.get("language", None)
        # Ensure row["text"] is a string, handle potential NaN from empty cells
        raw_text = str(row.get("text", "")) 
        text = f"[{lang}]" + raw_text if lang else raw_text
        
        # Ensure row["audio_path"] is treated as a string, handle potential NaN
        relative_audio_filename = str(row.get("audio_path","")) 
        
        if not relative_audio_filename:
            raise ValueError(f"Empty audio path in CSV at index {idx}, row: {row.to_dict()}")

        audio_path_str = relative_audio_filename.strip('"\'') # Clean up quotes

        # Construct absolute path
        if os.path.isabs(audio_path_str):
            audio_path = Path(audio_path_str)
        else:
            audio_path = self.audio_root / audio_path_str
        
        # print(f"Processing item {idx}: Attempting to load audio from: {audio_path}") # Debug print
            
        if not audio_path.is_file():
            error_msg = (f"Audio file not found at constructed path: '{audio_path}'.\n"
                         f"Derived from CSV audio_path entry: '{relative_audio_filename}', audio_root: '{self.audio_root}'.\n"
                         f"Problematic CSV row (index {idx}): {row.to_dict()}")
            # logger.error(error_msg) # if logger is defined
            print(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio file {audio_path} (index {idx}): {e}")
            raise
            
        if sr != 44100: # Assuming target SR is 44100
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        
        # Ensure waveform is (B, C, T) for DAC: (1, 1, num_samples) or (1, C, num_samples)
        if waveform.dim() == 1: # (T)
             waveform = waveform.unsqueeze(0).unsqueeze(0) 
        elif waveform.dim() == 2: # (C,T)
            waveform = waveform.unsqueeze(0) 

        with torch.no_grad():
            dac_device = next(self.dac_model.parameters()).device
            # Preprocess expects (B, 1, T_wav) or (B, C, T_wav)
            audio_tensor = self.dac_model.preprocess(waveform.to(dac_device), 44100)
            
            # Encode also expects input on model's device
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            # encoded is (B, D, T_codes), where D = num_quantizers
            # We need (T_codes, D) for returning from dataset, collate_fn will stack to (B, T_codes, D)
            encoded = encoded.squeeze(0).transpose(0, 1) 
        
        # Return waveform as (C, T) for consistency, collate_fn will handle batching
        return text, encoded, waveform.squeeze(0)
class HFDiaDataset(Dataset):
    def __init__(self, hf_dataset, config: DiaConfig, dac_model: dac.DAC):
        self.dataset = hf_dataset
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        lang = sample.get("language", None)
        text = f"[{lang}]" + sample["text"] if lang else sample["text"]
        audio_info = sample["audio_path"]
        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        sr = audio_info.get("sampling_rate", 44100)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        with torch.no_grad():
            audio_tensor = (
                self.dac_model.preprocess(waveform, 44100)
                .to(next(self.dac_model.parameters()).device)
            )
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform



class HFDiaIterDataset(torch.utils.data.IterableDataset):
    """Iterable wrapper for a HF streaming Dataset that has `audio.array` & `text`."""
    def __init__(self, hf_iterable, config: DiaConfig, dac_model: dac.DAC):
        super().__init__()
        self.dataset = hf_iterable
        self.config = config
        self.dac_model = dac_model

    def __iter__(self):
        for sample in self.dataset:
            lang = sample.get("language", None)
            text = f"[{lang}]" + sample["text"] if lang else sample["text"]
            audio_info = sample['audio']
            waveform = torch.tensor(audio_info['array'], dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            sr = audio_info.get('sampling_rate', 44100)
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
            with torch.no_grad():
                audio_tensor = (
                    self.dac_model.preprocess(waveform, 44100)
                    .to(next(self.dac_model.parameters()).device)
                )
                _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
                encoded = encoded.squeeze(0).transpose(0, 1)
            yield text, encoded, waveform
