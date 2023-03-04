import argparse
import json
import os
import pickle

import numpy as np
import torch

from .config import FLAGS
from .torch_model import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def convert_to_haiku(a, h, device):
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()
    hk_map = {}
    for a, b in generator.state_dict().items():
        print(a, b.shape)
        if a.startswith("conv_pre"):
            a = "generator/~/conv1_d"
        elif a.startswith("conv_post"):
            a = "generator/~/conv1_d_1"
        elif a.startswith("ups."):
            ii = a.split(".")[1]
            a = f"generator/~/ups_{ii}"
        elif a.startswith("resblocks."):
            _, x, y, z, _ = a.split(".")
            a = f"generator/~/res_block1_{x}/~/{y}_{z}"
        print(a, b.shape)
        if a not in hk_map:
            hk_map[a] = {}
        if len(b.shape) == 1:
            hk_map[a]["b"] = b.numpy()
        else:
            if "ups" in a:
                hk_map[a]["w"] = np.rot90(b.numpy(), k=1, axes=(0, 2))
            elif "conv" in a:
                hk_map[a]["w"] = np.swapaxes(b.numpy(), 0, 2)
            else:
                hk_map[a]["w"] = b.numpy()

    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(FLAGS.ckpt_dir / "hk_hifi.pickle", "wb") as f:
        pickle.dump(hk_map, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--config-file", required=True)
    a = parser.parse_args()

    config_file = a.config_file
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    device = torch.device("cpu")
    convert_to_haiku(a, h, device)


if __name__ == "__main__":
    main()
