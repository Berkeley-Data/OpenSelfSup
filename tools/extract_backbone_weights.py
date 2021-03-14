import torch
import argparse
import wandb
import os.path
from os import path


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('output', default=None, type=str, help='destination file name')
    parser.add_argument('checkpoint', default=None, type=str, help='W&B run id or local path (e.g., 3l4yg63k)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")

    checkpoint = args.checkpoint
    if path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        # if not checkpoint is not valid path, check for wandb
        projectid = "hpt2"
        restored_model = wandb.restore('latest.pth', run_path=f"{projectid}/{checkpoint}", replace=False)
        if restored_model is None:
            raise Exception(f"failed to load the model from runid or path: {checkpoint} ")
        ck = torch.load(restored_model.name, map_location=torch.device('cpu'))

    output_dict = dict(state_dict=dict(), author="OpenSelfSup")
    has_backbone = False
    for key, value in ck['state_dict'].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
