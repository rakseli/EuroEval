import os
import time
import subprocess
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument('--partition',help="slurm partition",default="small-g")
ap.add_argument('--time',help="slurm time",default="03:00:00")
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

def get_results():
    results = []
    try:
        with open("euroeval_benchmark_results_spesific_lrs.jsonl","r") as f:
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
    results = get_results()
    if results:
        for l in results:
            if l['dataset']==params['dataset'] and l['model'] == params['model']:
                print(f"Results for {params['dataset']}, model {params['model']} already exists, skipping...")
                return None
    running_jobs = get_running_job_names()

    if any([j for j in running_jobs if run_name in j]):
        print(f"Model {run_name} is currently running, not launching a job...")
        return None
    all_results = []
    with open("euroeval_benchmark_results_spesific_lrs.jsonl") as f:
        for l in f:
            try:
                all_results.append(json.loads(l))
            except json.JSONDecodeError as e:
                print(e)
    if any(d.get("dataset") == params['dataset'] and d.get("model") == params['model'] for d in all_results):
        print(f"Result for dataset {params['dataset']} and model {params['model']} already exists, skipping...")
        return None
    
    hp_search_command = "--no_hp_search"
    dataset_command = f"--dataset {params['dataset']}"
    validation_command = "--evaluate-test-split"
    lr_command = f"--learning_rate {params['best_lr']}"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --partition={args.partition}
#SBATCH --mem=64G
#SBATCH --time={args.time}
#SBATCH --account=project_462000963
#SBATCH --output=./logs/%x_%j.output
#SBATCH --error=./logs/%x_%j.error


#* fail the script if:
#* -e exit if any command [1] has a non-zero exit status
#* -u a reference to any variable you haven't previously defined - with the exceptions of $* and $@ - is an error
#* -o pipefail  If any command in a pipeline fails, that return code will be used as the return code of the whole pipeline
start_time=$(date +%s)
echo "Start time: $(date)"
#set -euo pipefail
module purge
export EBU_USER_PREFIX=/scratch/project_462000353/akselir/EasyBuild
module load CrayEnv
module load PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250619
set -euo pipefail
mkdir -p ./{run_name}
cd ./{run_name}
singularity exec $SIF euroeval --model {params['model']} {dataset_command} --language {params['language']} {lr_command} {validation_command} --debug --no-progress-bar {hp_search_command}
end_time=$(date +%s)
echo "End time: $(date)"
# Calculate total run time in seconds
runtime=$((end_time - start_time))

# Convert to hours using bc for floating point division
runtime_hours=$(echo "scale=4; $runtime / 3600" | bc)

echo "Total run time: $runtime_hours hours"
""" 

    return script_content

if __name__ == '__main__':
    args = ap.parse_args()
    runs = []
    with open("best_lrs.jsonl","r") as f:
        for l in f:
            runs.append(json.loads(l))
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