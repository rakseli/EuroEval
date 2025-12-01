import os
import time
import subprocess
import argparse
from launch_final_training_jobs import get_running_job_names

parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--model',default=None,help="model to be tested or none for multirun")
ap.add_argument('--language',help="language to be tested")
ap.add_argument('--dataset',nargs="*",default=None)
ap.add_argument('--lr',default=None)
ap.add_argument('--partition',help="slurm partition",default="small-g")
ap.add_argument('--time',help="slurm time",default="08:00:00")
ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
ap.add_argument('--test', action='store_true', help="launch one test job")


def create_slurm_scripts(model,running_jobs,args):
    """Creates a slurm script in right string format

    Args:
        args: CMD arguments
    Returns:
    - str: the script
    """
    finnish_datasets = "--dataset scandisent-fi --dataset turku-ner-fi --dataset tydiqa-fi --dataset scala-fi"
    swedish_datasets = "--dataset swerec --dataset scala-sv --dataset suc3 --dataset scandiqa-sv"
    english_datasets = "--dataset sst5 --dataset conll-en --dataset scala-en --dataset squad"
    if args.dataset is None:
        if args.language == "fi":
            dataset_to_be_used = finnish_datasets
        if args.language == "sv":
            dataset_to_be_used = swedish_datasets
        if args.language == "en":
            dataset_to_be_used = english_datasets
    
    if args.dataset is None and args.lr is None:
        run_name = f"euroeval-{model.replace('/','_')}-lang-{args.language}-all-datasets-hp-search"
        hp_search_command = "--hp_search"
        validation_command = "--evaluate-val-split"
        task_command = dataset_to_be_used
        lr_command = ""

    elif args.dataset is None and args.lr is not None:
        print("Dataset was not specified but specific lr was given, not launching a job.")
        return None

    elif args.dataset is not None and args.lr is not None:
        run_name = f"{model.replace('/','_')}-lang-{args.language}-task-{args.task}-lr-{args.lr}-final-training"
        hp_search_command = "--no_hp_search"
        task_command = f"--dataset {args.dataset}"
        validation_command = "--evaluate-test-split"
        lr_command = f"--learning_rate {args.lr}"



    if run_name in running_jobs:
        print(f"Job {run_name} is currently running, skipping...")
        return None
    

    script_content = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --partition={args.partition}
#SBATCH --mem=64G
#SBATCH --time={args.time}
#SBATCH --account=<project_number>
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
export EBU_USER_PREFIX=/scratch/<project_number>/akselir/EasyBuild
module load CrayEnv
module load PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250619
set -euo pipefail
mkdir -p {run_name}
cd {run_name}
singularity exec $SIF euroeval --model {model} {task_command} --language {args.language} {lr_command} {validation_command} --debug --no-progress-bar {hp_search_command}
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
    finnish_models = ["TurkuNLP/finnish-modernbert-base-short"]
    swedish_models = ["TurkuNLP/finnish-modernbert-base-short"]
    english_models = ["TurkuNLP/finnish-modernbert-base-short"]
    if args.model is None:       
        if args.language == "fi":
            models = finnish_models
        elif args.language == "sv":
            models = swedish_models

        elif args.language == "en":
            models = english_models

        else:
            raise ValueError(f"fi, sv or en should be used as language, {args.language} given")
    else:
        models = [args.model]
    job_count = 0
    running_jobs = get_running_job_names()
    for i,m in enumerate(models,start=1):
        command = create_slurm_scripts(m,running_jobs,args)
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
            running_jobs = get_running_job_names()    
            if args.test:
                break

    print(f"Launched {job_count} jobs")
