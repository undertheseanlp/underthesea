import pickle
from pathlib import Path

from tabulate import tabulate


def load_latest_ckpt(ckpt_dir: Path):
    ckpt = ckpt_dir / "duration_latest_ckpt.pickle"
    if not ckpt.exists():
        return None
    print("Loading latest checkpoint from file", ckpt)
    with open(ckpt, "rb") as f:
        dic = pickle.load(f)
    return dic["step"], dic["params"], dic["aux"], dic["rng"], dic["optim_state"]


def save_ckpt(step, params, aux, rng, optim_state, ckpt_dir: Path):
    dic = {
        "step": step,
        "params": params,
        "aux": aux,
        "rng": rng,
        "optim_state": optim_state,
    }
    with open(ckpt_dir / "duration_latest_ckpt.pickle", "wb") as f:
        pickle.dump(dic, f)


def print_flags(flags):
    values = [(k, v) for k, v in flags.items() if not k.startswith("_")]
    print(tabulate(values))
