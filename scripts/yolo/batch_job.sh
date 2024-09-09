#!/bin/bash

# Check if the project path and config file path are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <path_to_project> <path_to_config_file>"
    exit 1
fi

# Set the project path and config file path
project_path=$1
config_file=$2

# Export proxy settings
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Change to the project directory
cd "$project_path" || { echo "Failed to change directory to $project_path"; exit 1; }

# Activate the virtual environment
source ./.venv/bin/activate

# Install yq if it is not already installed
if ! command -v yq &> /dev/null
then
    echo "yq not found. Installing..."
    wget https://github.com/mikefarah/yq/releases/download/v4.6.3/yq_linux_amd64 -O ./.venv/bin/yq
    chmod +x ./.venv/bin/yq
fi

# Extract the values of the variables in the config using yq
time=$(yq e '.args.time' $config_file)
ntasks=$(yq e '.args.ntasks' $config_file)
job=$(yq e '.args.job' $config_file)
gres=$(yq e '.args.gres' $config_file)
C=$(yq e '.args.C' $config_file)
export=$(yq e '.args.job' $config_file)

if [[ $job == 'salloc' ]]; then
    # Actions to be performed when $job equals 'salloc'
    echo "salloc"
    salloc --gres=$gres --time=$time --ntasks=$ntasks -C=$C
else
    # Actions to be performed when $job is not equal to 'salloc'
    echo "sbatch"
    sbatch --gres=$gres --time=$time --ntasks=$ntasks -C $C $job
fi

squeue
