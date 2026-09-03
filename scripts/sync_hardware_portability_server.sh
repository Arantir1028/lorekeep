#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

profile="${1:-}"
if [[ -z "${profile}" || "${profile}" == "-h" || "${profile}" == "--help" ]]; then
  echo "usage: $0 {a100|rtx4090|4090|rtx5090|5090} [remote-host]" >&2
  echo >&2
  echo "optional environment variables:" >&2
  echo "  WAVESLICE_SYNC_LOCAL_ROOT" >&2
  echo "  WAVESLICE_SYNC_REMOTE_HOST" >&2
  echo "  WAVESLICE_SYNC_REMOTE_ROOT" >&2
  echo "  WAVESLICE_SYNC_SOURCE_RUN" >&2
  echo "  WAVESLICE_SYNC_HOST_LABEL" >&2
  [[ -n "${profile}" ]] && exit 0
  exit 2
fi

case "${profile}" in
  a100)
    gpu_profile="a100"
    ;;
  rtx4090|4090)
    gpu_profile="rtx4090"
    ;;
  rtx5090|5090)
    gpu_profile="rtx5090"
    ;;
  *)
    echo "unknown profile: ${profile}" >&2
    exit 2
    ;;
esac

local_root="${WAVESLICE_SYNC_LOCAL_ROOT:-${repo_root}}"
remote_host="${2:-${WAVESLICE_SYNC_REMOTE_HOST:-}}"
remote_root="${WAVESLICE_SYNC_REMOTE_ROOT:-waveslice-${gpu_profile}}"
host_label="${WAVESLICE_SYNC_HOST_LABEL:-${gpu_profile}}"
ratio_run="results/openworkload_ratio_sweep_lora8/ratio_sweep_20step_5models_a100_overnight_main"
source_run="${WAVESLICE_SYNC_SOURCE_RUN:-${repo_root}/${ratio_run}}"
remote_source_run="${remote_root}/${ratio_run}"

if [[ -z "${remote_host}" ]]; then
  echo "remote host is required as the second argument or WAVESLICE_SYNC_REMOTE_HOST" >&2
  exit 2
fi
if [[ ! -d "${local_root}" ]]; then
  echo "local repository root does not exist: ${local_root}" >&2
  exit 1
fi
if [[ ! -d "${source_run}" ]]; then
  echo "source run does not exist: ${source_run}" >&2
  exit 1
fi

ssh "${remote_host}" "mkdir -p '${remote_root}' '${remote_source_run}'"

rsync -az \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'results/' \
  "${local_root}/" "${remote_host}:${remote_root}/"

rsync -az "${source_run}/" "${remote_host}:${remote_source_run}/"

ssh "${remote_host}" "cd '${remote_root}' && python3 -m py_compile scripts/run_hardware_portability_experiment.py scripts/run_unified_distserve_cucumis_experiment.py scripts/run_cucumis_2a100_dispatch_sweep.py"

ssh "${remote_host}" "cd '${remote_root}' && python3 scripts/run_hardware_portability_experiment.py --gpu-profile '${gpu_profile}' --run-location server --host-label '${host_label}' --cuda-visible-devices 0,1 --dry-run"
