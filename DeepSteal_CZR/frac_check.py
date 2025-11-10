import torch
import argparse

def analyze_constraints(ckpt_path):
    # Load the checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cons = ckpt["constraints"]

    total = 0
    full = 0
    part = 0
    none = 0

    for name, c in cons.items():
        n = c["mask_full"].numel()
        total += n
        full += int(c["mask_full"].view(-1).sum().item())
        part += int(c["mask_part"].view(-1).sum().item())
        none += int(c["mask_none"].view(-1).sum().item())

    print(f"Results for: {ckpt_path}")
    print("Total elems:", total)
    print("Full  :", full, f"({full / total:.6f})")
    print("Part  :", part, f"({part / total:.6f})")
    print("None  :", none, f"({none / total:.6f})")

def main():
    parser = argparse.ArgumentParser(description="Analyze constraint masks from a checkpoint file.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the checkpoint file containing constraints.")
    args = parser.parse_args()

    analyze_constraints(args.ckpt)

if __name__ == "__main__":
    main()
