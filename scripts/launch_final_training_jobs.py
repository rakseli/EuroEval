import os
import time
import subprocess
import argparse
import json

DEFAULT_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "updated_files"))
APPEND_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "append_final_result.py"))

ap = argparse.ArgumentParser()
ap.add_argument('--data_path',help="directory containing best_lrs.jsonl and euroeval_benchmark_results_spesific_lrs.jsonl",default=DEFAULT_DATA_PATH)
ap.add_argument('--partition',help="slurm partition",default="gpumedium")
ap.add_argument('--time',help="slurm time",default="03:00:00")
ap.add_argument('--account',help="slurm billing project",default="project_2018556")
ap.add_argument('--container',help="apptainer image containing euroeval",default="/scratch/project_2018556/csc-containers/scripts/euroeval_container_20261112_161837.sif")
ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
ap.add_argument('--test', action='store_true', help="launch one test job")


def get_running_job_names():
    try:
        # Run the squeue command for current user
        result = subprocess.run(["squeue", "--me", "--Format=Name:200", "--noheader"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # Split the output into lines and strip whitespace
        job_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return job_names
    except subprocess.CalledProcessError as e:
        print(f"Error running squeue: {e.stderr}")
        return []

def get_results(data_path):
    results = []
    try:
        with open(os.path.join(data_path,"euroeval_benchmark_results_spesific_lrs.jsonl"),"r") as f:
            for l in f:
                try:
                    results.append(json.loads(l))
                except json.JSONDecodeError as e:
                    #print(e)
                    #print(f"Line: {l}")
                    pass
    
    except FileNotFoundError as e:
        print("Result file not yet found")
    
    return results
        
            

def create_slurm_scripts(params,args):
    """Creates a slurm script in right string format

    Args:
        args: CMD arguments
    Returns:
    - str: the script
    """

    run_name = f"{params['model'].replace('/','_')}-lang-{params['language']}-dataset-{params['dataset']}-lr-{params['best_lr']}-final-training"
    results = get_results(args.data_path)
    if any(d.get("dataset") == params['dataset'] and d.get("model") == params['model'] for d in results):
        print(f"Results for {params['dataset']}, model {params['model']} already exists, skipping...")
        return None
    running_jobs = get_running_job_names()

    if any([j for j in running_jobs if run_name in j]):
        print(f"Model {run_name} is currently running, not launching a job...")
        return None

    hp_search_command = "--no_hp_search"
    dataset_command = f"--dataset {params['dataset']}"
    validation_command = "--evaluate-test-split"
    lr_command = f"--learning_rate {params['best_lr']}"
    master_file = os.path.join(os.path.abspath(args.data_path), "euroeval_benchmark_results_spesific_lrs.jsonl")
    script_content = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --nodes=1
#* one GH200 GPU; on Roihu GPU partitions CPU cores (max 72) and memory (212 GiB) are fixed per reserved GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=./logs/%x_%j.output
#SBATCH --error=./logs/%x_%j.error


#* fail the script if:
#* -e exit if any command [1] has a non-zero exit status
#* -u a reference to any variable you haven't previously defined - with the exceptions of $* and $@ - is an error
#* -o pipefail  If any command in a pipeline fails, that return code will be used as the return code of the whole pipeline
set -euo pipefail
start_time=$(date +%s)
echo "Start time: $(date)"
SIF={args.container}
if [ ! -f "$SIF" ]; then
    echo "Container $SIF not found" >&2
    exit 1
fi
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export HF_HOME=/scratch/project_2018556/.cache/hf_cache

mkdir -p ./{run_name}
cd ./{run_name}
RUN_DIR="$PWD"
srun apptainer exec --nv --bind /scratch/{args.account} "$SIF" euroeval --model {params['model']} {dataset_command} --language {params['language']} {lr_command} {validation_command} --debug --no-progress-bar {hp_search_command}
#* only reached if euroeval succeeded (set -e); appends the result to the master file under a file lock
apptainer exec --bind /scratch/{args.account} "$SIF" python3 {APPEND_SCRIPT} --run_dir "$RUN_DIR" --master_file {master_file} --model {params['model']} --dataset {params['dataset']} --lr {params['best_lr']}
#* only reached if this run's result is in the master file (append exits 1 if a different result is there); the run dir holds the cached model, remove it to save disk
cd ..
if [ "$(basename "$RUN_DIR")" = "{run_name}" ]; then
    rm -rf -- "$RUN_DIR"
    echo "Removed $RUN_DIR"
fi
end_time=$(date +%s)
echo "End time: $(date)"
# Calculate total run time in seconds
runtime=$((end_time - start_time))

echo "Total run time: $((runtime / 3600))h $(((runtime % 3600) / 60))m $((runtime % 60))s"
"""

    return script_content

if __name__ == '__main__':
    args = ap.parse_args()
    runs = []
    with open(os.path.join(args.data_path,"best_lrs.jsonl"),"r") as f:
        for l in f:
            runs.append(json.loads(l))
    # slurm does not create the log directory and fails the job if it is missing
    os.makedirs("logs", exist_ok=True)
    job_count=0
    for i,m in enumerate(runs,start=1):
        command = create_slurm_scripts(m,args)
        if command is None:
            continue
        if args.dry_run:
            print(command)
        else:
            temp_file_name = f"{os.getcwd()}/temp_slurm_job.sh"
            with open(temp_file_name,"w") as temp_file:
                temp_file.write(command)
                # Submit the SLURM job using sbatch with the temporary file
            result=subprocess.run(["sbatch", temp_file_name], text=True)
            print(result)
            time.sleep(1)
            os.remove(temp_file_name)
            job_count+=1
            if args.test:
                break

    print(f"Launched {job_count} jobs")