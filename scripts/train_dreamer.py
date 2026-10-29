import argparse
import csv
import datetime
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DREAMER_DIR = ROOT / "third_party" / "dreamerv3-torch"
sys.path.insert(0, str(DREAMER_DIR))
sys.path.insert(0, str(ROOT))

import ruamel.yaml as yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.distributions.Distribution.set_default_validate_args(False)

import dreamer as dreamer_mod
import tools
from parallel import Damy

from dreamer_tmnf.wrapper import DreamerTMNF

CSV_FIELDS = ["episode", "total_env_steps", "sim_hours", "wall_hours", "grad_steps",
              "ep_return", "ep_len", "progress_m", "reason", "lap_time_s", "max_vram_gb"]


def load_config(remaining):
    loader = yaml.YAML(typ="safe")
    configs = loader.load((DREAMER_DIR / "configs.yaml").read_text())
    configs.update(loader.load((ROOT / "dreamer_tmnf" / "configs_tmnf.yaml").read_text()))

    def rec(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                rec(base[k], v)
            else:
                base[k] = v

    defaults = {}
    for name in ["defaults", "tmnf"]:
        rec(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        t = tools.args_type(value)
        parser.add_argument(f"--{key}", type=t, default=t(value))
    return parser.parse_args(remaining)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8479)
    p.add_argument("--game-speed", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=float, default=1e6)
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--max_wall_s", type=float, default=0)
    args, remaining = p.parse_known_args()
    config = load_config(remaining)
    config.seed = args.seed
    config.steps = int(args.steps)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    logdir = pathlib.Path(args.logdir or ROOT / "runs" / "dreamer" / f"{stamp}_{args.seed}")
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = logdir
    tools.set_seed_everywhere(config.seed)
    config.traindir = logdir / "train_eps"
    config.evaldir = logdir / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    config.eval_every = int(config.eval_every)
    config.log_every = int(config.log_every)
    with open(logdir / "config.json", "w") as f:
        json.dump({**vars(config), "port": args.port, "game_speed": args.game_speed, "cmdline": sys.argv}, f, indent=1, default=str)

    step0 = dreamer_mod.count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step0)
    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    t_start = time.time()
    holder = {"agent": None, "ep": 0}
    csv_path = logdir / "episodes.csv"
    new = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new:
        writer.writeheader()
    jsonl = open(logdir / "episodes.jsonl", "a")

    def on_episode(info, ret, length):
        holder["ep"] += 1
        agent = holder["agent"]
        finished = bool(info.get("finished"))
        row = {
            "episode": holder["ep"],
            "total_env_steps": info.get("total_env_steps"),
            "sim_hours": info.get("total_sim_ms", 0) / 3.6e6,
            "wall_hours": (time.time() - t_start) / 3600.0,
            "grad_steps": agent._update_count if agent is not None else 0,
            "ep_return": ret,
            "ep_len": length,
            "progress_m": info.get("progress_m"),
            "reason": info.get("reason"),
            "lap_time_s": info.get("sim_time_s") if finished else None,
            "max_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        }
        writer.writerow(row)
        csv_file.flush()
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        logger.scalar("ep/progress_m", row["progress_m"] or 0.0)
        logger.scalar("ep/sim_hours", row["sim_hours"])
        logger.scalar("ep/wall_hours", row["wall_hours"])
        logger.scalar("ep/grad_steps", row["grad_steps"])
        logger.scalar("ep/finished", 1.0 if finished else 0.0)

    env = DreamerTMNF(port=args.port, time_limit=config.time_limit, on_episode=on_episode, speed=args.game_speed)
    train_envs = [Damy(env)]
    config.num_actions = env.action_space.shape[0]

    random_actor = tools.OneHotDist(torch.zeros(config.num_actions).repeat(1, 1))

    def random_agent(o, d, s):
        action = random_actor.sample()
        return {"action": action, "logprob": random_actor.log_prob(action)}, None

    prefill = max(0, config.prefill - step0)
    print(f"Prefill {prefill}", flush=True)
    state = tools.simulate(random_agent, train_envs, train_eps, config.traindir, logger,
                           limit=config.dataset_size, steps=prefill)
    logger.step += prefill * config.action_repeat

    dataset = dreamer_mod.make_dataset(train_eps, config)
    agent = dreamer_mod.Dreamer(env.observation_space, env.action_space, config, logger, dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    holder["agent"] = agent
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Params total: {n_params / 1e6:.2f}M", flush=True)
    with open(logdir / "params.json", "w") as f:
        json.dump({"total_params": n_params}, f)
    if (logdir / "latest.pt").exists():
        ck = torch.load(logdir / "latest.pt")
        agent.load_state_dict(ck["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, ck["optims_state_dict"])
        agent._should_pretrain._once = False

    torch.cuda.reset_peak_memory_stats()
    while agent._step < config.steps:
        state = tools.simulate(agent, train_envs, train_eps, config.traindir, logger,
                               limit=config.dataset_size, steps=config.eval_every, state=state)
        torch.save({"agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent)},
                   logdir / "latest.pt")
        if args.max_wall_s and time.time() - t_start > args.max_wall_s:
            break
    env.close()


if __name__ == "__main__":
    main()
