import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.ticker as ticker

def generate_batch_size_plot(results, output_path):
    """Generate batch size performance plot with all metrics"""
    fig, axs = plt.subplots(4, 2, figsize=(15, 24))  # 4x2布局
    fig.suptitle('Performance by Batch Size', fontsize=16)
    
    batch_sizes = results['batch_sizes']
    
    # 统一设置x轴刻度
    def set_batch_size_xticks(ax):
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels(batch_sizes)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # 1. Total Completion Time
    axs[0, 0].plot(batch_sizes, results['serial_total_times'], 'o-', color='skyblue', label='Serial')
    axs[0, 0].plot(batch_sizes, results['parallel_total_times'], 'o-', color='orange', label='Parallel')
    axs[0, 0].set_title('Total Completion Time')
    axs[0, 0].set_xlabel('Batch Size')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].grid(True, which="both", ls="--")
    axs[0, 0].legend()
    set_batch_size_xticks(axs[0, 0])
    
    # 2. Average Request Time
    axs[0, 1].plot(batch_sizes, results['serial_avg_request_times'], 'o-', color='skyblue', label='Serial')
    axs[0, 1].plot(batch_sizes, results['parallel_avg_request_times'], 'o-', color='orange', label='Parallel')
    axs[0, 1].set_title('Average Request Time')
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('Time (seconds)')
    axs[0, 1].grid(True, which="both", ls="--")
    axs[0, 1].legend()
    set_batch_size_xticks(axs[0, 1])
    
    # 3. Throughput
    axs[1, 0].plot(batch_sizes, results['serial_throughputs'], 'o-', color='skyblue', label='Serial')
    axs[1, 0].plot(batch_sizes, results['parallel_throughputs'], 'o-', color='orange', label='Parallel')
    axs[1, 0].set_title('Throughput')
    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Samples per second')
    axs[1, 0].grid(True, which="both", ls="--")
    axs[1, 0].legend()
    set_batch_size_xticks(axs[1, 0])
    
    # 4. GPU Utilization (if available)
    if results['serial_gpu_utils']:
        axs[1, 1].plot(batch_sizes, results['serial_gpu_utils'], 'o-', color='skyblue', label='Serial')
        axs[1, 1].plot(batch_sizes, results['parallel_gpu_utils'], 'o-', color='orange', label='Parallel')
        axs[1, 1].set_title('GPU Utilization')
        axs[1, 1].set_xlabel('Batch Size')
        axs[1, 1].set_ylabel('Utilization (%)')
        axs[1, 1].set_ylim(0, 100)
        axs[1, 1].grid(True, which="both", ls="--")
        axs[1, 1].legend()
        set_batch_size_xticks(axs[1, 1])
    else:
        axs[1, 1].text(0.5, 0.5, 'No GPU Data Available', ha='center', va='center')
        axs[1, 1].axis('off')
    
    # 5. Memory Utilization (if available)
    if results['serial_mem_utils']:
        axs[2, 0].plot(batch_sizes, results['serial_mem_utils'], 'o-', color='skyblue', label='Serial')
        axs[2, 0].plot(batch_sizes, results['parallel_mem_utils'], 'o-', color='orange', label='Parallel')
        axs[2, 0].set_title('Memory Utilization')
        axs[2, 0].set_xlabel('Batch Size')
        axs[2, 0].set_ylabel('Utilization (%)')
        axs[2, 0].set_ylim(0, 100)
        axs[2, 0].grid(True, which="both", ls="--")
        axs[2, 0].legend()
        set_batch_size_xticks(axs[2, 0])
    else:
        axs[2, 0].text(0.5, 0.5, 'No Memory Data Available', ha='center', va='center')
        axs[2, 0].axis('off')
    
    # 6. Memory Used (if available)
    if results['serial_mem_used']:
        axs[2, 1].plot(batch_sizes, results['serial_mem_used'], 'o-', color='skyblue', label='Serial')
        axs[2, 1].plot(batch_sizes, results['parallel_mem_used'], 'o-', color='orange', label='Parallel')
        axs[2, 1].set_title('Memory Used')
        axs[2, 1].set_xlabel('Batch Size')
        axs[2, 1].set_ylabel('Memory (MB)')
        axs[2, 1].grid(True, which="both", ls="--")
        axs[2, 1].legend()
        set_batch_size_xticks(axs[2, 1])
    else:
        axs[2, 1].text(0.5, 0.5, 'No Memory Used Data Available', ha='center', va='center')
        axs[2, 1].axis('off')
    
    # 7. Model Actual Execution Time (combined)
    axs[3, 0].plot(batch_sizes, results['serial_model1_times'], 'o-', color='skyblue', label='Serial Model1')
    axs[3, 0].plot(batch_sizes, results['parallel_model1_times'], 'o-', color='orange', label='Parallel Model1')
    axs[3, 0].plot(batch_sizes, results['serial_model2_times'], 'o--', color='skyblue', label='Serial Model2')
    axs[3, 0].plot(batch_sizes, results['parallel_model2_times'], 'o--', color='orange', label='Parallel Model2')
    axs[3, 0].set_title('Model Actual Execution Time')
    axs[3, 0].set_xlabel('Batch Size')
    axs[3, 0].set_ylabel('Time (seconds)')
    axs[3, 0].grid(True, which="both", ls="--")
    axs[3, 0].legend()
    set_batch_size_xticks(axs[3, 0])
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"Batch size performance plot saved to: {output_path}")

# Read CSV data
csv_path = 'example/1-128/hf_two_models_dual_stream.csv'
results = {
    'batch_sizes': [],
    'serial_total_times': [],
    'parallel_total_times': [],
    'serial_avg_request_times': [],
    'parallel_avg_request_times': [],
    'serial_throughputs': [],
    'parallel_throughputs': [],
    'serial_model1_times': [],
    'parallel_model1_times': [],
    'serial_model2_times': [],
    'parallel_model2_times': [],
    'serial_gpu_utils': [],
    'parallel_gpu_utils': [],
    'serial_mem_utils': [],
    'parallel_mem_utils': [],
    'serial_mem_used': [],
    'parallel_mem_used': []
}

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Skip header
    for row in reader:
        if row:  # Skip empty rows
            results['batch_sizes'].append(int(row[0]))
            results['serial_total_times'].append(float(row[1]))
            results['parallel_total_times'].append(float(row[2]))
            results['serial_avg_request_times'].append(float(row[3]))
            results['parallel_avg_request_times'].append(float(row[4]))
            results['serial_throughputs'].append(float(row[5]))
            results['parallel_throughputs'].append(float(row[6]))
            results['serial_model1_times'].append(float(row[7]))
            results['parallel_model1_times'].append(float(row[8]))
            results['serial_model2_times'].append(float(row[9]))
            results['parallel_model2_times'].append(float(row[10]))
            results['serial_gpu_utils'].append(float(row[11]))
            results['parallel_gpu_utils'].append(float(row[12]))
            results['serial_mem_utils'].append(float(row[13]))
            results['parallel_mem_utils'].append(float(row[14]))
            results['serial_mem_used'].append(float(row[15]))
            results['parallel_mem_used'].append(float(row[16]))

# Generate plot
output_path = 'example/1-128/batch_performance_plot.png'
generate_batch_size_plot(results, output_path)
print("图表已保存为 " + output_path) 