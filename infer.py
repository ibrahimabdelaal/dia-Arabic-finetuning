import argparse
import os
import random
import numpy as np
import soundfile as sf
import torch
import re
import sys
from typing import List, Tuple

# Add current directory and mantoq to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
mantoq_path = os.path.join(current_dir, 'mantoq')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if mantoq_path not in sys.path:
    sys.path.insert(0, mantoq_path)

try:
    from mantoq import g2p
except ImportError as e:
    print(f"Error importing mantoq: {e}")
    print("Make sure mantoq is installed or available in the current directory")
    # Try alternative import paths
    try:
        sys.path.insert(0, os.path.join(current_dir, '..'))
        from mantoq import g2p
        print("Successfully imported mantoq from parent directory")
    except ImportError:
        try:
            from mantoq.mantoq import g2p
            print("Successfully imported g2p from mantoq.mantoq")
        except ImportError:
            print("Failed to import mantoq. Please check installation.")
            sys.exit(1)

from dia.model import Dia


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_ascii(phonemes):
    """Convert phonemes to ASCII representation."""
    mapping = {
        'ʔ': "'", 'ʕ': 'E', 'ː': '', '_+_': '_dbl_',
    }
    processed = []
    for ph in phonemes:
        if 'ː' in ph:
            base = ph.replace('ː', '')
            processed.append(base * 2)
        else:
            processed.append(mapping.get(ph, ph))
    return processed


def clean_text(text):
    """Clean Arabic text by removing unwanted characters."""
    text = re.sub(r"[^\w\s\u064B-\u0652]", "", text)
    return text.strip()


def preprocess_arabic_text_for_tts(input_text):
    """
    Converts Arabic text to ASCII-phoneme sequence with [ar][S1] tag.
    """
    try:
        _, phonemes = g2p(input_text, add_tashkeel=False)
        ascii_phonemes = convert_to_ascii(phonemes)
        
        # Clean up problematic patterns
        phoneme_str = " ".join(ascii_phonemes)
        # # Fix inconsistent dbl patterns
        # phoneme_str = re.sub(r'\*dbl_', '*dbl*', phoneme_str)
        # # Remove excessive consecutive dbl tokens
        # phoneme_str = re.sub(r'(\*dbl\*\s+){3,}', '*dbl* ', phoneme_str)
        # # Clean up extra spaces
        # phoneme_str = re.sub(r'\s+', ' ', phoneme_str).strip()
        
        return phoneme_str
    except Exception as e:
        print(f"[WARN] Preprocessing failed: {e}")
        return "[ar][S1]"


def smart_chunk_arabic_text(text: str, max_chars: int = 200) -> List[str]:
    """
    Intelligently chunk Arabic text based on punctuation and character limits.
    
    Args:
        text: Input Arabic text
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Define Arabic punctuation marks for splitting
    arabic_punctuation = ['،', '؛', '؟', '!', '.', ':', '\n', '؟', '٪']
    
    chunks = []
    current_chunk = ""
    
    # First, split by major punctuation
    sentences = re.split(r'[.؟!؛\n]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If the sentence itself is longer than max_chars, split by commas
        if len(sentence) > max_chars:
            sub_parts = re.split(r'[،,]+', sentence)
            
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                    
                # If still too long, split by word boundaries
                if len(part) > max_chars:
                    words = part.split()
                    temp_chunk = ""
                    
                    for word in words:
                        # If adding this word exceeds limit, save current chunk
                        if len(temp_chunk + " " + word) > max_chars and temp_chunk:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk += (" " + word) if temp_chunk else word
                    
                    if temp_chunk:
                        if current_chunk and len(current_chunk + " " + temp_chunk) <= max_chars:
                            current_chunk += " " + temp_chunk
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = temp_chunk
                else:
                    # Check if we can add this part to current chunk
                    if current_chunk and len(current_chunk + " " + part) > max_chars:
                        chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        current_chunk += (" " + part) if current_chunk else part
        else:
            # Check if we can add this sentence to current chunk
            if current_chunk and len(current_chunk + " " + sentence) > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
    
    # Add the last chunk if exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out empty chunks and ensure minimum chunk size
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 10]
    
    return chunks


def crossfade_audio(audio1: np.ndarray, audio2: np.ndarray, 
                   crossfade_duration: float = 0.1, sample_rate: int = 44100) -> np.ndarray:
    """
    Crossfade between two audio arrays.
    
    Args:
        audio1: First audio array
        audio2: Second audio array
        crossfade_duration: Duration of crossfade in seconds
        sample_rate: Sample rate of audio
        
    Returns:
        Crossfaded audio array
    """
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # Ensure we don't exceed the length of either audio
    crossfade_samples = min(crossfade_samples, len(audio1), len(audio2))
    
    if crossfade_samples == 0:
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    
    # Apply crossfade
    audio1_end = audio1[-crossfade_samples:] * fade_out
    audio2_start = audio2[:crossfade_samples] * fade_in
    
    # Combine
    crossfaded_section = audio1_end + audio2_start
    
    # Concatenate the full result
    result = np.concatenate([
        audio1[:-crossfade_samples],
        crossfaded_section,
        audio2[crossfade_samples:]
    ])
    
    return result


def combine_audio_chunks(audio_chunks: List[np.ndarray], 
                        sample_rate: int = 44100,
                        crossfade_duration: float = 0.05,
                        pause_duration: float = 0.2) -> np.ndarray:
    """
    Combine multiple audio chunks with crossfading and pauses.
    
    Args:
        audio_chunks: List of audio arrays
        sample_rate: Sample rate of audio
        crossfade_duration: Duration of crossfade between chunks
        pause_duration: Duration of pause between chunks
        
    Returns:
        Combined audio array
    """
    if not audio_chunks:
        return np.array([])
    
    if len(audio_chunks) == 1:
        return audio_chunks[0]
    
    combined = audio_chunks[0]
    pause_samples = int(pause_duration * sample_rate)
    pause_audio = np.zeros(pause_samples)
    
    for i in range(1, len(audio_chunks)):
        # Add a small pause before crossfading
        combined = np.concatenate([combined, pause_audio])
        
        # Crossfade with the next chunk
        combined = crossfade_audio(combined, audio_chunks[i], 
                                 crossfade_duration, sample_rate)
    
    return combined


def generate_audio_for_chunks(model, chunks: List[str], args) -> List[np.ndarray]:
    """
    Generate audio for each text chunk with proper error handling and memory management.
    
    Args:
        model: Dia model instance
        chunks: List of preprocessed text chunks
        args: Command line arguments
        
    Returns:
        List of generated audio arrays
    """
    audio_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Generating audio for chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        
        # Clear GPU cache before each generation if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        max_retries = 3
        success = False
        
        for retry in range(max_retries):
            try:
                # Add some debugging info
                print(f"  Text length: {len(chunk)} characters (attempt {retry + 1})")
                
                # Adjust generation parameters for problematic chunks
                current_max_tokens = args.max_tokens
                current_temperature = args.temperature
                current_cfg_scale = args.cfg_scale
                
                # For retries, try more conservative settings
                if retry > 0:
                    current_max_tokens = min(args.max_tokens, 2000 - retry * 500)
                    current_temperature = max(0.1, args.temperature - retry * 0.1)
                    current_cfg_scale = max(1.0, args.cfg_scale - retry * 0.5)
                    print(f"  Retry {retry + 1} with: max_tokens={current_max_tokens}, temp={current_temperature:.2f}, cfg={current_cfg_scale:.1f}")
                
                # Ensure minimum token count
                if current_max_tokens < 500:
                    current_max_tokens = 500
                
                output_audio = model.generate(
                    text=chunk,
                    audio_prompt_path=args.audio_prompt,
                    max_tokens=current_max_tokens,
                    cfg_scale=current_cfg_scale,
                    temperature=current_temperature,
                    top_p=args.top_p,
                )
                
                # Validate output
                if output_audio is None or len(output_audio) == 0:
                    print(f"⚠ Warning: Chunk {i+1} returned empty audio on attempt {retry + 1}")
                    if retry == max_retries - 1:
                        # Last attempt failed, use silence
                        silence = np.zeros(int(2.0 * 44100))  # 2 seconds of silence
                        audio_chunks.append(silence)
                        success = True
                    continue
                else:
                    # Convert to numpy array if not already
                    if torch.is_tensor(output_audio):
                        output_audio = output_audio.cpu().numpy()
                    
                    # Additional validation - check if audio has reasonable length
                    if len(output_audio) < 1000:  # Less than ~0.02 seconds
                        print(f"⚠ Warning: Chunk {i+1} generated very short audio ({len(output_audio)} samples)")
                        if retry == max_retries - 1:
                            # Pad with silence if too short
                            padding = np.zeros(int(1.0 * 44100))
                            output_audio = np.concatenate([output_audio, padding])
                    
                    audio_chunks.append(output_audio)
                    print(f"✓ Chunk {i+1} generated successfully (duration: {len(output_audio)/44100:.2f}s)")
                    success = True
                    break
                
            except Exception as e:
                print(f"✗ Error generating chunk {i+1} (attempt {retry + 1}): {type(e).__name__}: {e}")
                
                if retry == max_retries - 1:
                    print(f"  All retries failed for chunk {i+1}, using silence")
                    # Try to continue with silence instead of failing completely
                    silence = np.zeros(int(2.0 * 44100))  # 2 seconds of silence
                    audio_chunks.append(silence)
                    success = True
                else:
                    print(f"  Retrying chunk {i+1}...")
                    # Optional: try to recover by reinitializing model state
                    try:
                        if hasattr(model, 'reset_state'):
                            model.reset_state()
                        elif hasattr(model, 'eval'):
                            model.eval()
                    except:
                        pass
                    
                    # Small delay before retry
                    import time
                    time.sleep(0.5)
        
        if not success:
            print(f"Critical: Could not generate audio for chunk {i+1}")
            silence = np.zeros(int(2.0 * 44100))
            audio_chunks.append(silence)
        
        # Small delay between chunks to prevent resource conflicts
        import time
        time.sleep(0.2)
    
    return audio_chunks


def main():
    parser = argparse.ArgumentParser(description="Generate audio using the Dia model with smart chunking.")
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="مِنْ جِهَةٍ ثَانِيَةٍ أَشَارَتْ إلَى أَنَّ أَهَمَّ الْمَحَطَّاتِ الْفَنِّيَّةِ فِي حَيَاتِهَا كَانَتْ وُقُوفُهَا عَلَى أَهَمِّ الْمَسَارِحِ كَمَسْرَحِ تِرْيَانُونَ فِي بَارِيسَ وَمَسَارِحِ عِدَّةٍ فِي أَمِيرِكَا",
        help="Input Arabic text for speech generation."
    )
    
    parser.add_argument(
        "--text-file",
        type=str,
        help="Path to text file containing Arabic text to convert."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str, 
        required=True, 
        help="Path to save the generated audio file (e.g., output.wav)."
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum characters per chunk (default: 200)."
    )
    
    parser.add_argument(
        "--crossfade-duration",
        type=float,
        default=0.05,
        help="Crossfade duration between chunks in seconds (default: 0.05)."
    )
    
    parser.add_argument(
        "--pause-duration",
        type=float,
        default=0.2,
        help="Pause duration between chunks in seconds (default: 0.2)."
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="nari-labs/Dia-1.6B",
        help="Hugging Face repository ID (e.g., nari-labs/Dia-1.6B).",
    )
    parser.add_argument(
        "--local-paths", 
        action="store_true", 
        help="Load model from local config and checkpoint files."
    )

    parser.add_argument(
        "--config", "-c",
        type=str, 
        default="dia/config_inference.json", 
        help="Path to local config.json file (default: dia/config_inference.json)."
    )
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str, 
        help="Path to local model checkpoint .pth file (required if --local-paths is set)."
    )
    parser.add_argument(
        "--audio-prompt", 
        type=str, 
        default=None, 
        help="Path to an optional audio prompt WAV file for voice cloning."
    )

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum number of audio tokens to generate (defaults to config value).",
    )
    gen_group.add_argument(
        "--cfg-scale", 
        type=float, 
        default=3.0, 
        help="Classifier-Free Guidance scale (default: 3.0)."
    )
    gen_group.add_argument(
        "--temperature", 
        type=float, 
        default=0.4, 
        help="Sampling temperature (higher is more random, default: 0.4)."
    )
    gen_group.add_argument(
        "--top-p", 
        type=float, 
        default=0.95, 
        help="Nucleus sampling probability (default: 0.95)."
    )

    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for reproducibility."
    )
    infra_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., 'cuda', 'cpu', default: auto).",
    )

    args = parser.parse_args()

    # Validation for local paths
    if args.local_paths:
        if not args.config:
            parser.error("--config is required when --local-paths is set.")
        if not args.checkpoint:
            parser.error("--checkpoint is required when --local-paths is set.")
        if not os.path.exists(args.config):
            parser.error(f"Config file not found: {args.config}")
        if not os.path.exists(args.checkpoint):
            parser.error(f"Checkpoint file not found: {args.checkpoint}")
    
    # Get input text
    if args.text_file:
        if not os.path.exists(args.text_file):
            parser.error(f"Text file not found: {args.text_file}")
        with open(args.text_file, 'r', encoding='utf-8') as f:
            input_text = f.read().strip()
    else:
        input_text = args.text
    
    if not input_text:
        parser.error("No input text provided.")
            
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Determine device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Step 1: Clean and chunk the text
    print("Step 1: Cleaning and chunking Arabic text...")
    cleaned_text = clean_text(input_text)
    text_chunks = smart_chunk_arabic_text(cleaned_text, args.chunk_size)
    print(f"Text divided into {len(text_chunks)} chunks:")
    for i, chunk in enumerate(text_chunks):
        print(f"  Chunk {i+1}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")

    # Step 2: Convert each chunk to phonemes
    print("\nStep 2: Converting chunks to phonemes...")
    phoneme_chunks = []
    failed_chunks = 0
    
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{len(text_chunks)}...")
        try:
            phoneme_chunk = preprocess_arabic_text_for_tts(chunk)
            
            # Validate phoneme conversion
            if not phoneme_chunk or phoneme_chunk == "[ar][S1]":
                print(f"⚠ Warning: Chunk {i+1} phoneme conversion failed, using fallback")
                phoneme_chunk = f"[ar][S1] {chunk}"  # Fallback to original text
                failed_chunks += 1
            
            phoneme_chunks.append(phoneme_chunk)
            print(f"  → {phoneme_chunk[:80]}{'...' if len(phoneme_chunk) > 80 else ''}")
            
        except Exception as e:
            print(f"✗ Error processing chunk {i+1}: {e}")
            # Use fallback
            phoneme_chunk = f"[ar][S1] {chunk}"
            phoneme_chunks.append(phoneme_chunk)
            failed_chunks += 1
    
    if failed_chunks > 0:
        print(f"⚠ {failed_chunks} chunks had phoneme conversion issues, using fallbacks")

    # Step 3: Load model
    print("\nStep 3: Loading model...")
    if args.local_paths:
        print(f"Loading from local paths: config='{args.config}', checkpoint='{args.checkpoint}'")
        try:
            model = Dia.from_local(args.config, args.checkpoint, device=device)
        except Exception as e:
            print(f"Error loading local model: {e}")
            exit(1)
    else:
        print(f"Loading from Hugging Face Hub: repo_id='{args.repo_id}'")
        try:
            model = Dia.from_pretrained(args.repo_id, device=device)
        except Exception as e:
            print(f"Error loading model from Hub: {e}")
            exit(1)
    print("Model loaded successfully.")

    # Step 4: Generate audio for each chunk
    print("\nStep 4: Generating audio for each chunk...")
    audio_chunks = generate_audio_for_chunks(model, phoneme_chunks, args)

    # Step 5: Combine audio chunks with crossfading
    print("\nStep 5: Combining audio chunks with crossfading...")
    sample_rate = 44100
    
    try:
        combined_audio = combine_audio_chunks(
            audio_chunks,
            sample_rate=sample_rate,
            crossfade_duration=args.crossfade_duration,
            pause_duration=args.pause_duration
        )
        
        # Step 6: Save the final audio
        print(f"\nStep 6: Saving final audio to {args.output}...")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        sf.write(args.output, combined_audio, sample_rate)
        
        print(f"✓ Audio successfully saved to {args.output}")
        print(f"Final audio duration: {len(combined_audio) / sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Error during audio combination or saving: {e}")
        exit(1)


if __name__ == "__main__":
    main()
