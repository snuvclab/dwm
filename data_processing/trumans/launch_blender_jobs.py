import os
import subprocess
import time
from collections import deque

# === User Configuration ===
RECORDINGS_PATH = "../nas1/public_dataset/trumans/Recordings_blend"
SCRIPT_PATH = "blender_ego_rgb_depth.py"
NUM_GPUS = 8
MAX_PARALLEL_PROCESSES = NUM_GPUS
# ===========================

# (1) Discover all .blend files recursively
blend_jobs = []
for root, dirs, files in os.walk(RECORDINGS_PATH):
    for file in files:
        if file.endswith(".blend"):
            blend_jobs.append(os.path.join(root, file))

print(f"Found {len(blend_jobs)} .blend files.")

# (2) Track running processes with GPU assignment
running_processes = {}  # GPU_ID -> list of processes
available_gpus = deque(range(NUM_GPUS))  # Queue of available GPUs

# (3) Dispatch jobs with proper GPU distribution
for idx, blend_file in enumerate(blend_jobs):
    # Wait for a GPU to become available
    while len(available_gpus) == 0:
        # Check if any processes have finished
        finished_gpus = []
        for gpu_id, processes in running_processes.items():
            # Remove finished processes
            processes[:] = [p for p in processes if p.poll() is None]
            # If no processes left on this GPU, mark it as available
            if len(processes) == 0:
                finished_gpus.append(gpu_id)
        
        # Add finished GPUs back to available queue
        for gpu_id in finished_gpus:
            available_gpus.append(gpu_id)
            del running_processes[gpu_id]
        
        if len(available_gpus) == 0:
            print("All GPUs busy, waiting...")
            time.sleep(5)  # Wait before checking again
    
    # Get next available GPU
    gpu_id = available_gpus.popleft()
    
    cmd = [
        "blender",
        "--background",
        blend_file,
        "--python",
        SCRIPT_PATH
    ]

    # Assign environment variable to restrict GPU usage
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Launching Blender for: {os.path.basename(blend_file)}")
    proc = subprocess.Popen(cmd, env=env)
    
    # Track this process on the assigned GPU
    if gpu_id not in running_processes:
        running_processes[gpu_id] = []
    running_processes[gpu_id].append(proc)

# (4) Wait for all remaining processes to finish
print("Waiting for all processes to complete...")
while running_processes:
    finished_gpus = []
    for gpu_id, processes in running_processes.items():
        # Remove finished processes
        processes[:] = [p for p in processes if p.poll() is None]
        # If no processes left on this GPU, mark it as finished
        if len(processes) == 0:
            finished_gpus.append(gpu_id)
            print(f"GPU {gpu_id} completed all jobs")
    
    # Remove finished GPUs
    for gpu_id in finished_gpus:
        del running_processes[gpu_id]
    
    if running_processes:
        print(f"Still waiting for {sum(len(procs) for procs in running_processes.values())} processes on {len(running_processes)} GPUs...")
        time.sleep(10)

print("All Blender render jobs completed.")

