"""Remix an existing dataset workload with a new arrival layout and schedule."""

from __future__ import annotations

import argparse
import json
import os

from experiments.build_dataset_workload import _arrival_order, _assign_poisson_arrivals


def _load(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description='Remix an existing workload prefix into a new arrival stream.')
    parser.add_argument('--src-prefix', required=True)
    parser.add_argument('--out-prefix', required=True)
    parser.add_argument('--arrival-mode', choices=['burst', 'poisson'], default='poisson')
    parser.add_argument('--phase1-arrival-rate', type=float, default=6.0)
    parser.add_argument('--phase2-arrival-rate', type=float, default=6.0)
    parser.add_argument('--arrival-seed', type=int, default=7)
    parser.add_argument('--phase1-arrival-layout', choices=['grouped', 'mixed', 'beneficiary_rich'], default='beneficiary_rich')
    parser.add_argument('--phase2-arrival-layout', choices=['grouped', 'mixed', 'beneficiary_rich'], default='beneficiary_rich')
    parser.add_argument('--phase1-early-short-frac', type=float, default=0.25)
    parser.add_argument('--phase2-early-short-frac', type=float, default=0.20)
    parser.add_argument('--phase1-post-long-short-bias', type=float, default=0.70)
    parser.add_argument('--phase2-post-long-short-bias', type=float, default=0.60)
    args = parser.parse_args()

    src_req = f'{args.src_prefix}_requests.json'
    src_lora = f'{args.src_prefix}_lora_requests.json'
    src_meta = f'{args.src_prefix}_meta.json'
    reqs = _load(src_req)
    lora_reqs = _load(src_lora)
    meta = _load(src_meta) if os.path.exists(src_meta) else {}

    reqs = _arrival_order(
        reqs,
        seed=int(args.arrival_seed) + 17,
        layout=str(args.phase1_arrival_layout),
        early_short_frac=float(args.phase1_early_short_frac),
        post_long_short_bias=float(args.phase1_post_long_short_bias),
    )
    lora_reqs = _arrival_order(
        lora_reqs,
        seed=int(args.arrival_seed) + 1017,
        layout=str(args.phase2_arrival_layout),
        early_short_frac=float(args.phase2_early_short_frac),
        post_long_short_bias=float(args.phase2_post_long_short_bias),
    )

    if args.arrival_mode == 'poisson':
        reqs = _assign_poisson_arrivals(reqs, rate_per_s=float(args.phase1_arrival_rate), seed=int(args.arrival_seed))
        lora_reqs = _assign_poisson_arrivals(lora_reqs, rate_per_s=float(args.phase2_arrival_rate), seed=int(args.arrival_seed) + 1009)
    else:
        reqs = [dict(item, arrival_offset_s=0.0) for item in reqs]
        lora_reqs = [dict(item, arrival_offset_s=0.0) for item in lora_reqs]

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    req_path = f'{args.out_prefix}_requests.json'
    lora_path = f'{args.out_prefix}_lora_requests.json'
    meta_path = f'{args.out_prefix}_meta.json'
    with open(req_path, 'w', encoding='utf-8') as f:
        json.dump(reqs, f, ensure_ascii=False, indent=2)
    with open(lora_path, 'w', encoding='utf-8') as f:
        json.dump(lora_reqs, f, ensure_ascii=False, indent=2)
    meta.update({
        'arrival_mode': args.arrival_mode,
        'phase1_arrival_layout': args.phase1_arrival_layout,
        'phase2_arrival_layout': args.phase2_arrival_layout,
        'phase1_arrival_rate': args.phase1_arrival_rate,
        'phase2_arrival_rate': args.phase2_arrival_rate,
        'phase1_request_count': len(reqs),
        'phase2_request_count': len(lora_reqs),
        'phase1_last_arrival_s': max((float(item.get('arrival_offset_s', 0.0)) for item in reqs), default=0.0),
        'phase2_last_arrival_s': max((float(item.get('arrival_offset_s', 0.0)) for item in lora_reqs), default=0.0),
        'source_workload_prefix': args.src_prefix,
    })
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'[Saved] {req_path}')
    print(f'[Saved] {lora_path}')
    print(f'[Saved] {meta_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
