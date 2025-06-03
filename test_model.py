import torch
from dia.layers import DiaModel  # Adjust import if needed
from dia.config import DiaConfig
import os # For path manipulation

def check_model_dtypes(model):
    print("Checking model parameter dtypes:")
    for name, param in model.named_parameters():
        print(f"Param: {name} | dtype: {param.dtype}")

    print("\nChecking model buffer dtypes:")
    for name, buffer in model.named_buffers():
        print(f"Buffer: {name} | dtype: {buffer.dtype}")

def main():
    # Set your config and checkpoint paths here:
    config_path = "/home/ubuntu/work/dia-finetuning/dia/config.json"
    original_checkpoint_path = "/home/ubuntu/work/dia-finetuning/.cpkts/dia_finetune_cv/ckpt_step6000.pth"

    # --- Define path for the new float32 checkpoint ---
    checkpoint_dir = os.path.dirname(original_checkpoint_path)
    checkpoint_basename = os.path.basename(original_checkpoint_path)
    # Example: ckpt_step6000_float32.pth
    new_checkpoint_filename = f"{os.path.splitext(checkpoint_basename)[0]}_float32.pth"
    float32_checkpoint_path = os.path.join(checkpoint_dir, new_checkpoint_filename)
    # --- ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config and instantiate model
    dia_cfg = DiaConfig.load(config_path)
    model = DiaModel(dia_cfg) # Instantiate on CPU

    # Load original checkpoint (map to CPU first for manipulation)
    print(f"Loading original checkpoint from: {original_checkpoint_path}")
    state_dict = torch.load(original_checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print("Original checkpoint loaded.")

    # Check dtypes before conversion (optional, for verification)
    print("\n--- Dtypes before conversion ---")
    check_model_dtypes(model)

    # Convert all model parameters and buffers to float32
    print("\nConverting model to float32...")
    model.float() # This operation is in-place for parameters
    print("Model converted to float32.")

    # Check dtypes after conversion (on CPU)
    print("\n--- Dtypes after conversion (on CPU) ---")
    check_model_dtypes(model)

    # --- Save the float32 model state_dict ---
    # It's generally recommended to save the state_dict of a model on the CPU
    # to ensure maximum portability when loading it later on any device.
    print(f"\nSaving float32 model state_dict to: {float32_checkpoint_path}")
    torch.save(model.state_dict(), float32_checkpoint_path)
    print("Float32 model state_dict saved successfully.")
    # --- ---

    # Now move the model to the desired device for use (if needed)
    model = model.to(device)
    print(f"\nModel moved to {device} for potential immediate use.")

    # Final check of dtypes on the target device (optional)
    print("\n--- Dtypes after moving to device ---")
    check_model_dtypes(model)


if __name__ == "__main__":
    main()