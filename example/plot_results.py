import matplotlib.pyplot as plt
import numpy as np

# 结果字符串
result_str = (
    "Serial: total=279.8277s, avg_task=0.1749s, throughput=5.72 req/s, gpu_util=38.9%, mem_util=17.9%, mem_used=14631.7 MiB | "
    "Streams: total=210.4488s, avg_task=0.1315s, throughput=7.60 req/s, gpu_util=43.2%, mem_util=21.4%, mem_used=17549.4 MiB | "
    "BigBatch: total=146.9647s, avg_task=0.0919s, throughput=10.89 req/s, gpu_util=40.9%, mem_util=21.9%, mem_used=17911.0 MiB"
)

# 提取数值
modes = []
totals = []
avg_tasks = []
throughputs = []
gpu_utils = []
mem_utils = []

sections = result_str.split(' | ')
for section in sections:
    if not section.strip():
        continue
    # 分割模式名和键值对
    mode, data_str = section.split(':', 1)
    mode = mode.strip()
    modes.append(mode)
    
    # 分割键值对
    data_parts = [p.strip() for p in data_str.split(',')]
    
    # 解析每个键值对
    total = float(data_parts[0].split('=')[1].split('s')[0])
    avg_task = float(data_parts[1].split('=')[1].split('s')[0])
    throughput = float(data_parts[2].split('=')[1].split()[0])
    gpu_util = float(data_parts[3].split('=')[1].split('%')[0])
    mem_util = float(data_parts[4].split('=')[1].split('%')[0])
    
    totals.append(total)
    avg_tasks.append(avg_task)
    throughputs.append(throughput)
    gpu_utils.append(gpu_util)
    mem_utils.append(mem_util)

# 创建图形
fig, ax = plt.subplots(1, 5, figsize=(25, 5))

# Total Time
ax[0].bar(modes, totals, color=['skyblue', 'orange', 'green'])
ax[0].set_title('Total Time (s)')
ax[0].set_ylabel('Seconds')
for i, v in enumerate(totals):
    ax[0].text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')

# Avg Task Time
ax[1].bar(modes, avg_tasks, color=['skyblue', 'orange', 'green'])
ax[1].set_title('Avg Task Time (s)')
ax[1].set_ylabel('Seconds')
for i, v in enumerate(avg_tasks):
    ax[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# Throughput
ax[2].bar(modes, throughputs, color=['skyblue', 'orange', 'green'])
ax[2].set_title('Throughput (req/s)')
ax[2].set_ylabel('req/s')
for i, v in enumerate(throughputs):
    ax[2].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')

# GPU Utilization
ax[3].bar(modes, gpu_utils, color=['skyblue', 'orange', 'green'])
ax[3].set_title('GPU Utilization (%)')
ax[3].set_ylabel('%')
for i, v in enumerate(gpu_utils):
    ax[3].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

# Memory Utilization
ax[4].bar(modes, mem_utils, color=['skyblue', 'orange', 'green'])
ax[4].set_title('Memory Utilization (%)')
ax[4].set_ylabel('%')
for i, v in enumerate(mem_utils):
    ax[4].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('hf_performance_comparison.png')
plt.show()

print("图表已保存为 hf_performance_comparison.png") 