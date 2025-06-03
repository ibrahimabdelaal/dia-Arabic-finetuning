import json
import csv
from pathlib import Path
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed

# CONFIG
data_dir = Path("/home/ubuntu/work/dia-finetuning/output_audio_tashkeelav2_test/content/output_audio_tashkeelav2_test/")
output_dir = Path("tashkeela_audio_output")
output_csv = "tashkeela_metadata.csv"

# Define duration thresholds (in seconds)
min_chunk = 3.0  # Minimum desired chunk duration
max_chunk = 20.0 # Maximum desired chunk duration

# Number of worker processes
num_workers = 32  # Adjust based on your CPU cores

output_dir.mkdir(parents=True, exist_ok=True)

def process_file(json_file):
    audio_file = json_file.with_suffix(".mp3")
    if not audio_file.exists():
        print(f"❌ Missing audio: {audio_file.name}")
        return []

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support for both JSON structures: new nested "alignment" or original top-level keys.
    if "alignment" in data:
        alignment = data["alignment"]
        chars = alignment.get("characters", [])
        start_times = alignment.get("character_start_times_seconds", [])
        end_times = alignment.get("character_end_times_seconds", [])
    else:
        chars = data.get("characters", [])
        start_times = data.get("character_start_times_seconds", [])
        end_times = data.get("character_end_times_seconds", [])

    # Load audio file
    audio = AudioSegment.from_mp3(audio_file)

    # Ensure lengths match
    if not (len(chars) == len(start_times) == len(end_times)):
        print(f"❌ Mismatch in lengths for {json_file.name}: chars={len(chars)}, starts={len(start_times)}, ends={len(end_times)}")
        return []

    # === Step 1: Build word list ===
    words = []
    word_chars = []
    word_start = None

    for i, c in enumerate(chars):
        if word_start is None:
            word_start = start_times[i]

        word_chars.append(c)

        # When we hit a boundary (space or punctuation), we finalize a word.
        if c == " " or c in "،؟.!؟":
            word_end = end_times[i]
            words.append({
                "text": ''.join(word_chars).strip(),
                "start": word_start,
                "end": word_end
            })
            word_chars = []
            word_start = None

    # Add last word if leftover
    if word_chars:
        word_end = end_times[-1]
        words.append({
            "text": ''.join(word_chars).strip(),
            "start": word_start,
            "end": word_end
        })

    # === Step 2: New Chunking Logic ===
    # Group words so that each chunk is as long as possible without exceeding max_chunk,
    # but if possible, each exported chunk should be at least min_chunk seconds.
    results = []
    current_chunk = []      # List of words in the current chunk.
    chunk_start_time = None
    chunk_idx = 0

    for word in words:
        # Start a new chunk if needed.
        if chunk_start_time is None:
            chunk_start_time = word["start"]
            current_chunk = [word]
            continue

        # Compute potential duration if we add this word.
        new_duration = word["end"] - chunk_start_time

        if new_duration <= max_chunk:
            # If adding the word stays within max_chunk, simply add it.
            current_chunk.append(word)
        else:
            # If adding this word would exceed max_chunk:
            current_duration = current_chunk[-1]["end"] - chunk_start_time
            if current_duration >= min_chunk:
                # Export current chunk if it's long enough.
                chunk_end_time = current_chunk[-1]["end"]
                chunk_audio = audio[int(chunk_start_time * 1000):int(chunk_end_time * 1000)]
                chunk_text = ' '.join([w["text"] for w in current_chunk if w["text"] not in "،؟.!؟ "])
                chunk_filename = f"test_hamed{json_file.stem}_chunk{chunk_idx}.wav"
                chunk_path = output_dir / chunk_filename
                chunk_audio.export(chunk_path, format="wav")
                results.append([str(chunk_path), chunk_text])
                chunk_idx += 1

                # Start a new chunk with the current word.
                chunk_start_time = word["start"]
                current_chunk = [word]
            else:
                # If current chunk is still too short (< min_chunk), add the word even if it exceeds max_chunk.
                current_chunk.append(word)

    # Export any remaining words as the final chunk if it meets the minimum duration.
    if current_chunk:
        chunk_end_time = current_chunk[-1]["end"]
        current_duration = chunk_end_time - chunk_start_time
        if current_duration >= min_chunk:
            chunk_audio = audio[int(chunk_start_time * 1000):int(chunk_end_time * 1000)]
            chunk_text = ' '.join([w["text"] for w in current_chunk if w["text"] not in "،؟.!؟ "])
            chunk_filename = f"hamed_{json_file.stem}_chunk{chunk_idx}.wav"
            chunk_path = output_dir / chunk_filename
            chunk_audio.export(chunk_path, format="wav")
            results.append([str(chunk_path), chunk_text])
        else:
            print(f"⏩ Skipped final chunk in {json_file.name} (duration {current_duration:.2f}s is less than {min_chunk}s)")

    return results

def main():
    json_files = sorted(data_dir.glob("*.json"))
    print(f"📦 Found {len(json_files)} JSON files to process")

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_file, json_file): json_file for json_file in json_files}

        for future in as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"✅ Processed: {json_file.name}")
            except Exception as e:
                print(f"❌ Error processing {json_file.name}: {e}")

    # Write all results to CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow(["audio_path", "text"])
        for result in all_results:
            writer.writerow(result)

    print(f"\n✅ Done. Chunked audio saved to: {output_dir}")
    print(f"📄 Metadata saved to: {output_csv}")

if __name__ == "__main__":
    main()
