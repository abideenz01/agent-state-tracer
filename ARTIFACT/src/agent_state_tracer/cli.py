"""Command line entry point.

    python -m agent_state_tracer analyze  <spans.jsonl | otlp.json> [--json out.json]
    python -m agent_state_tracer simulate --out spans.jsonl [--inputs 400 --replays 5 --seed 7]
    python -m agent_state_tracer evaluate [--out-dir DIR]
    python -m agent_state_tracer collector-config [--rate 0.05]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

log = logging.getLogger("agent_state_tracer")


def _rate(text: str) -> float:
    v = float(text)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError("rate must be between 0 and 1")
    return v


def _positive(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent_state_tracer", description=__doc__.split("\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="check completeness and explain path splits in an export")
    a.add_argument("path")
    a.add_argument("--json", dest="json_out", help="also write the report as JSON")
    a.add_argument("--top", type=_positive, default=5)

    s = sub.add_parser("simulate", help="write a synthetic span export")
    s.add_argument("--out", required=True)
    s.add_argument("--inputs", type=_positive, default=400)
    s.add_argument("--replays", type=_positive, default=5)
    s.add_argument("--seed", type=int, default=7)

    e = sub.add_parser("evaluate", help="run the synthetic evaluation")
    e.add_argument("--out-dir", help="write eval_results.json and eval_results.md here")
    e.add_argument("--inputs", type=_positive, default=400)
    e.add_argument("--replays", type=_positive, default=5)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--rate", type=_rate, default=0.05, help="tail baseline probabilistic rate")

    cc = sub.add_parser("collector-config", help="print a Collector tail_sampling snippet")
    cc.add_argument("--rate", type=_rate, default=0.05)
    cc.add_argument("--decision-wait", default="30s")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.cmd == "analyze":
            from .analyze import analyze, format_report
            from .loader import load_spans

            spans = load_spans(args.path)
            log.debug("loaded %d spans from %s", len(spans), args.path)
            rep = analyze(spans, top=args.top)
            print(format_report(rep))
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as fh:
                    json.dump(rep, fh, indent=2, sort_keys=True)
            return 0 if rep["incomplete_traces"] == 0 else 3
        if args.cmd == "simulate":
            from .simulate import simulate, write_jsonl

            if args.replays < 2:
                raise ValueError("--replays must be at least 2")
            res = simulate(keys=args.inputs, replays=args.replays, seed=args.seed)
            write_jsonl(res.records, args.out)
            log.info("wrote %d spans for %d runs to %s", len(res.records), len(res.truths), args.out)
            return 0
        if args.cmd == "evaluate":
            from .evaluate import evaluate, to_markdown

            rep = evaluate(keys=args.inputs, replays=args.replays, seed=args.seed, tail_rate=args.rate)
            md = to_markdown(rep)
            print(md)
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                with open(os.path.join(args.out_dir, "eval_results.json"), "w", encoding="utf-8") as fh:
                    json.dump(rep, fh, indent=2, sort_keys=True)
                with open(os.path.join(args.out_dir, "eval_results.md"), "w", encoding="utf-8") as fh:
                    fh.write(md)
            return 0
        if args.cmd == "collector-config":
            from .sampler import collector_config

            print(collector_config(args.rate, args.decision_wait), end="")
            return 0
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
