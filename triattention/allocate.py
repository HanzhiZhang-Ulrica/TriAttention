from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from triattention.maps import INF


def tile_counts(w8, w4, rows):
    span = np.maximum(np.arange(rows) - 1, 0)
    n8 = np.minimum(span, w8 - 1)
    n4 = np.minimum(span, w4 - 1) - n8
    return int(n8.sum()), int(np.maximum(n4, 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocate band configurations to heads at several cost targets.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--c4", type=float, default=0.905)
    parser.add_argument("--fractions", default="0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95")
    args = parser.parse_args()
    cal = json.loads(Path(args.calibration).read_text())
    configs = [tuple(c) for c in cal["configs"]]
    layers = sorted(cal["errors"], key=int)
    err = np.array([cal["errors"][layer] for layer in layers])
    rows = (args.length + 63) // 64
    full = tile_counts(INF, INF, rows)[0]
    cost = np.array([(n8 + args.c4 * n4) / full for n8, n4 in (tile_counts(w8, w4, rows) for w8, w4 in configs)])
    frontier = []
    for lam in np.concatenate([[0.0], np.logspace(-9, 1, 400)]):
        choice = np.argmin(err + lam * cost[None, None, :], axis=2)
        frontier.append((float(cost[choice].mean()), float(np.take_along_axis(err, choice[..., None], 2).sum()), choice))
    maps = {}
    for fraction in sorted(float(x) for x in args.fractions.split(",")):
        point_cost, point_error, choice = min((f for f in frontier if f[0] <= fraction), key=lambda f: f[1])
        bands = {layer: [list(configs[c]) for c in choice[i]] for i, layer in enumerate(layers)}
        maps[str(fraction)] = dict(cost=point_cost, error=point_error, bands=bands)
        print(f"cost <= {fraction}: cost {point_cost:.3f}, error {point_error:.4g}")
    result = dict(format="triattention-candidates-v1", model_type=cal["model_type"], length=args.length, c4=args.c4,
                  maps=maps)
    Path(args.output).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
