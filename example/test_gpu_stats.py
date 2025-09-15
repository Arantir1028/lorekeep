import argparse
import time
import subprocess


def collect_gpu_stats(device_index: int = 1) -> tuple[float, float, float]:
    try:
        output = subprocess.check_output(
            f"nvidia-smi -i {device_index} -q -d UTILIZATION,MEMORY",
            shell=True,
            text=True
        )
        lines = output.splitlines()
        util = mem_used = mem_total = 0.0
        in_fb_memory = False
        in_util = False

        for line in lines:
            stripped = line.strip()

            # 进入 FB Memory Usage 部分
            if stripped.startswith("FB Memory Usage"):
                in_fb_memory = True
                continue
            if in_fb_memory and stripped.startswith("Total"):
                mem_total = float(stripped.split(":")[1].strip().split()[0])
            elif in_fb_memory and stripped.startswith("Used"):
                mem_used = float(stripped.split(":")[1].strip().split()[0])
                in_fb_memory = False  # 拿到Used就可以退出这个block

            # 进入 Utilization 部分
            if stripped.startswith("Utilization"):
                in_util = True
                continue
            if in_util and stripped.startswith("Gpu"):
                util = float(stripped.split(":")[1].strip().split("%")[0])
                in_util = False  # GPU拿到就够了

        mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0.0
        return util, mem_util, mem_used
    except Exception as e:
        print(f"GPU stats collection failed: {e}")
        return 0.0, 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Test GPU stats collection")
    parser.add_argument("--device_index", type=int, default=1, help="GPU device index for nvidia-smi")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to collect")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between samples in seconds")
    args = parser.parse_args()

    print(f"Testing GPU stats for device {args.device_index}, {args.samples} samples every {args.interval}s...")

    for i in range(args.samples):
        util, mem_util, mem_used = collect_gpu_stats(args.device_index)
        print(f"Sample {i+1}: GPU Util: {util:.1f}%, Mem Util: {mem_util:.1f}%, Mem Used: {mem_used:.1f} MiB")
        time.sleep(args.interval)


if __name__ == "__main__":
    main() 