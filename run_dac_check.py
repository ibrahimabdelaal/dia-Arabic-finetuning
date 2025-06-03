import argparse
import os
import torch
import torchaudio
import soundfile as sf # For saving, if torchaudio has issues with certain formats/ G711 codec error

# Attempt to import the Dia model class.
# Adjust this import if your project structure is different.
try:
    from dia.model import Dia
except ImportError:
    print("ERROR: Could not import the 'Dia' model class from 'dia.model'.")
    print("Please ensure this script is run from a location where the 'dia' package is accessible,")
    print("or that the 'dia' package is installed in your Python environment.")
    exit(1)

# Global device (will be set in main_dac_check)
device = None

def check_dac_reconstruction(dac_model_instance, audio_file_path, output_dir="dac_reconstruction_check"):
    """
    Encodes an audio file using the DAC model and then decodes it to check reconstruction quality.
    Saves the original (resampled) and reconstructed audio files.
    """
    global device # Use the globally set device

    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]

    if not hasattr(dac_model_instance, 'sample_rate'):
        print(f"Error: dac_model_instance does not have a 'sample_rate' attribute.")
        print("Cannot determine target sample rate. Please check your DAC model object.")
        return
        
    target_sample_rate = dac_model_instance.sample_rate
    
    print(f"\nProcessing: {audio_file_path}")
    print(f"DAC model expected sample rate: {target_sample_rate} Hz")
    print(f"Using device: {device}")

    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
        waveform = waveform.to(device)
    except Exception as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return

    # Resample if necessary
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate).to(device)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Convert to mono
    if waveform.shape[0] > 1:
        print("Converting to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Ensure waveform is in shape (B, C, T) -> (1, 1, num_samples) for DAC
    if waveform.dim() == 1: # (T)
        waveform = waveform.unsqueeze(0).unsqueeze(0) # (1,1,T)
    elif waveform.dim() == 2: # (C, T) assuming C=1 after mono
        waveform = waveform.unsqueeze(0) # (1, C, T)
    
    original_save_path = os.path.join(output_dir, f"{base_filename}_original_resampled_to_{target_sample_rate}hz.wav")
    try:
        torchaudio.save(original_save_path, waveform.squeeze(0).cpu(), target_sample_rate)
        print(f"Saved original (resampled) audio to: {original_save_path}")
    except Exception as e: # Fallback to soundfile if torchaudio fails (e.g. due to specific save types)
        print(f"torchaudio.save failed: {e}. Trying soundfile.")
        sf.write(original_save_path, waveform.squeeze(0).cpu().numpy().T, target_sample_rate)
        print(f"Saved original (resampled) audio using soundfile to: {original_save_path}")

    print(f"Input audio tensor shape for DAC: {waveform.shape}")

    try:
        with torch.no_grad():
            # Encode the audio
            # For descript-audio-codec, encode usually returns: (z, codes, latents, obj_quantized, metadata)
            encode_output = dac_model_instance.encode(waveform)
            
            if not (isinstance(encode_output, tuple) and len(encode_output) >= 4):
                print(f"Unexpected DAC encode output format: {type(encode_output)}. Cannot perform detailed check.")
                return

            z, codes, _, obj_quantized, _ = encode_output
            
            print(f"  Encoded continuous latent 'z' shape: {z.shape}")
            print(f"  Encoded discrete 'codes' shape: {codes.shape}") # (B, NumQuantizers, T_codes)
            print(f"  Encoded 'obj_quantized' (input to DAC decoder) shape: {obj_quantized.shape}")

            # Decode from continuous latent z (best possible reconstruction from DAC's encoder part)
            reconstructed_audio_from_z = dac_model_instance.decode(z)
            path_z = os.path.join(output_dir, f"{base_filename}_reconstructed_from_z.wav")
            torchaudio.save(path_z, reconstructed_audio_from_z.squeeze(0).cpu(), target_sample_rate)
            print(f"  Reconstructed audio (from continuous z) saved to: {path_z}")

            # Decode from quantized representation (this is closer to what Dia model's predictions would target)
            if obj_quantized is not None:
                reconstructed_audio_from_obj_q = dac_model_instance.decode(obj_quantized)
                path_obj_q = os.path.join(output_dir, f"{base_filename}_reconstructed_from_quantized.wav")
                torchaudio.save(path_obj_q, reconstructed_audio_from_obj_q.squeeze(0).cpu(), target_sample_rate)
                print(f"  Reconstructed audio (from quantized latents) saved to: {path_obj_q}")
            else:
                print("  Could not test reconstruction from quantized latents ('obj_quantized' was None).")

    except Exception as e:
        print(f"Error during DAC encode/decode for {audio_file_path}: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 40)


def main_dac_check():
    global device # Allow main_dac_check to set the global device

    parser = argparse.ArgumentParser(description="Check DAC tokenization for audio samples using Dia's DAC model.")
    
    model_load_group = parser.add_argument_group("Dia Model Loading Options")
    model_load_group.add_argument("--repo-id", type=str, default="nari-labs/Dia-1.6B", help="Hugging Face repo ID for Dia model.")
    model_load_group.add_argument("--local-paths", action="store_true", help="Load Dia model from local config and checkpoint files.")
    model_load_group.add_argument("--config", type=str, help="Path to local Dia config.json file (required if --local-paths is set).")
    model_load_group.add_argument("--checkpoint", type=str, help="Path to local Dia model checkpoint .pth file (required if --local-paths is set).")
    
    dac_check_group = parser.add_argument_group("DAC Check Options")
    dac_check_group.add_argument("--audio-files", type=str, nargs='+', required=True, help="Path(s) to audio file(s) to check (e.g., your Arabic samples).")
    dac_check_group.add_argument("--output-dir", type=str, default="dac_reconstruction_output", help="Directory to save reconstructed audio files.")
    
    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()

    device = torch.device(args.device) # Set global device
    print(f"Using device: {device}")

    # Load Dia model to access its dac_model component
    print("Loading Dia model to access its DAC component...")
    dia_pipeline = None
    if args.local_paths:
        if not (args.config and args.checkpoint):
            parser.error("--config and --checkpoint are required when using --local-paths for Dia model loading.")
        if not os.path.exists(args.config): parser.error(f"Dia config file not found: {args.config}")
        if not os.path.exists(args.checkpoint): parser.error(f"Dia checkpoint file not found: {args.checkpoint}")
        try:
            dia_pipeline = Dia.from_local(args.config, args.checkpoint, device=device)
        except Exception as e:
            print(f"Error loading local Dia model from config='{args.config}', checkpoint='{args.checkpoint}': {e}")
            exit(1)
    else:
        try:
            dia_pipeline = Dia.from_pretrained(args.repo_id, device=device)
        except Exception as e:
            print(f"Error loading Dia model from Hugging Face Hub (repo_id='{args.repo_id}'): {e}")
            exit(1)
    
    if not dia_pipeline or not hasattr(dia_pipeline, 'dac_model') or dia_pipeline.dac_model is None:
        print("Error: Could not access dac_model from the loaded Dia pipeline. Ensure 'dia_pipeline.dac_model' exists.")
        exit(1)
        
    dac_model_instance = dia_pipeline.dac_model
    dac_model_instance.eval() # Set DAC model to evaluation mode
    print("Dia model and its DAC component loaded successfully.")

    # Call the check_dac_reconstruction function for each audio file
    for audio_file_path in args.audio_files:
        check_dac_reconstruction(dac_model_instance, audio_file_path, args.output_dir)
    
    print("\nDAC reconstruction check finished.")
    print(f"Please check the files saved in the '{args.output_dir}' directory.")

if __name__ == "__main__":
    main_dac_check()