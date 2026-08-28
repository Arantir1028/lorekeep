#!/usr/bin/env bash
set -euo pipefail

profile="${1:-}"
if [[ -z "${profile}" ]]; then
  echo "usage: $0 {a100|rtx4090|4090|rtx5090|5090}" >&2
  exit 2
fi

case "${profile}" in
  a100)
    gpu_profile="a100"
    local_root="/home/onceas/Arantir/CUCUMIS-lorekeep-a100"
    remote_host="matpool-cucumis-a100"
    remote_root="/root/CUCUMIS-lorekeep-a100"
    host_label="matpool-a100"
    ;;
  rtx4090|4090)
    gpu_profile="rtx4090"
    local_root="/home/onceas/Arantir/CUCUMIS-lorekeep-4090"
    remote_host="matpool-cucumis-4090"
    remote_root="/root/CUCUMIS-lorekeep-4090"
    host_label="matpool-4090"
    ;;
  rtx5090|5090)
    gpu_profile="rtx5090"
    local_root="/home/onceas/Arantir/CUCUMIS-lorekeep-5090"
    remote_host="matpool-cucumis-5090"
    remote_root="/root/CUCUMIS-lorekeep-5090"
    host_label="matpool-5090"
    ;;
  *)
    echo "unknown profile: ${profile}" >&2
    exit 2
    ;;
esac

source_run="/home/onceas/Arantir/CUCUMIS-lorekeep/lorekeep/results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main"
remote_source_run="${remote_root}/results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main"

ssh "${remote_host}" "mkdir -p '${remote_root}' '${remote_source_run}'"

rsync -az \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'results/hardware_portability_a100_4090_5090/' \
  "${local_root}/" "${remote_host}:${remote_root}/"

rsync -az "${source_run}/" "${remote_host}:${remote_source_run}/"

ssh "${remote_host}" "cd '${remote_root}' && python3 -m py_compile scripts/run_hardware_portability_experiment.py scripts/run_unified_distserve_cucumis_experiment.py scripts/run_cucumis_2a100_dispatch_sweep.py"

ssh "${remote_host}" "cd '${remote_root}' && python3 scripts/run_hardware_portability_experiment.py --gpu-profile '${gpu_profile}' --run-location server --host-label '${host_label}' --cuda-visible-devices 0,1 --dry-run"
