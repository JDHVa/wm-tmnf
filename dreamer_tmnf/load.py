import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DREAMER_DIR = ROOT / "third_party" / "dreamerv3-torch"
for _p in (str(DREAMER_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ruamel.yaml as yaml
import tools


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
