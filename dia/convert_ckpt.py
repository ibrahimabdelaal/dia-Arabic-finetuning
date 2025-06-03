import argparse
import torch
from dia.layers import DiaModel  # adjust your import if needed
from dia.config import DiaConfig

def convert_checkpoint(input_ckpt: str, output_ckpt: str, config_path: str):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Reconstruct exactly the same compiled model you saved
    dia_cfg = DiaConfig.load(config_path)
    model = DiaModel(dia_cfg).to(device)
   # model = model.half()
    #model = torch.compile(model, backend="inductor")
    # def convert_model_to_bf16(model):
    # for param in model.parameters():
    #     param.data = param.data.to(torch.bfloat16)
    # for buffer_name, buffer in model.named_buffers():
    #     model.register_buffer(buffer_name, buffer.to(torch.bfloat16))
    # return model

    # 2) Load your compiled/half checkpoint
    state = torch.load(input_ckpt, map_location=device)
    model.load_state_dict(state)

    # 3) Un-wrap to the original nn.Module
    orig = getattr(model, "_orig_mod", None) or getattr(model, "__wrapped__", None) or model

    # 4) Cast all params & buffers back to float32
    orig.float()

    # 5) Save its clean, float32 state_dict
    torch.save(orig.state_dict(), output_ckpt)
    print(f"Saved normal FP32 checkpoint to {output_ckpt}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a compiled/half-precision checkpoint back to a standard FP32 state_dict."
    )
    parser.add_argument(
        "--input-ckpt", "-i",
        required=True,
        help="Path to the half-precision compiled checkpoint (.pth) to load"
    )
    parser.add_argument(
        "--output-ckpt", "-o",
        required=True,
        help="Path where the FP32 state_dict will be saved"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to your DiaConfig JSON file"
    )

    args = parser.parse_args()
    convert_checkpoint(args.input_ckpt, args.output_ckpt, args.config)

if __name__ == "__main__":
    main()