import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "dreamerv3-torch"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.distributions.Distribution.set_default_validate_args(False)

import dreamer as dreamer_mod
import tools
from dreamer_tmnf.wrapper import DreamerTMNF
from train_dreamer import load_config


def main():
    eps_dir = pathlib.Path(sys.argv[1])
    extra = sys.argv[2:]
    config = load_config(extra)
    config.num_actions = 9
    config.logdir = pathlib.Path("runs/bench/tmp")
    config.logdir.mkdir(parents=True, exist_ok=True)
    env = DreamerTMNF()
    eps = tools.load_episodes(eps_dir)
    logger = tools.Logger(config.logdir, 0)
    dataset = dreamer_mod.make_dataset(eps, config)
    agent = dreamer_mod.Dreamer(env.observation_space, env.action_space, config, logger, dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    n = sum(p.numel() for p in agent.parameters())
    for _ in range(5):
        agent._train(next(dataset))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    k = 40
    for _ in range(k):
        agent._train(next(dataset))
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(json.dumps({"params_M": n / 1e6, "grad_per_s": k / dt, "vram_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                      "extra": extra}))


if __name__ == "__main__":
    main()
