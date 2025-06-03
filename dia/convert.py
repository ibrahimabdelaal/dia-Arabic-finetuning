import argparse
import torch
from dia.layers import DiaModel
from dia.config import DiaConfig

def convert_checkpoint(input_ckpt: str, output_ckpt: str, config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and model
    dia_cfg = DiaConfig.load(config_path)
    model = DiaModel(dia_cfg)
    
    # Load half-precision checkpoint
    state = torch.load(input_ckpt, map_location=device)
    model.load_state_dict(state)

    # Convert all model params and buffers to float32
    for param in model.__dict__.get('params', []):
        param.data = param.data.float()
    for buffer in model.__dict__.get('buffers', []):
        buffer.data = buffer.data.float()

    # If model is a standard nn.Module, use this:
    try:
        model = model.float()
    except AttributeError:
        pass  # ignore if not supported

    # Save converted state dict
    torch.save(model.state_dict(), output_ckpt)
    print(f"Saved FP32 checkpoint to {output_ckpt}")

def main():
    parser = argparse.ArgumentParser(description="Convert a checkpoint to float32.")
    parser.add_argument("--input-ckpt", "-i", required=True, help="Path to input checkpoint (.pth)")
    parser.add_argument("--output-ckpt", "-o", required=True, help="Path to output checkpoint (.pth)")
    parser.add_argument("--config", "-c", required=True, help="Path to Dia config.json")

    args = parser.parse_args()
    convert_checkpoint(args.input_ckpt, args.output_ckpt, args.config)

if __name__ == "__main__":
    main()
