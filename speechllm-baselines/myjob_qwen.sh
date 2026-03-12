#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -f script_file [-t time_limit] [-g gpu_count] [-n name]"
    exit 1
}

# Default values
time_limit="48:00:00"
gpu_count=1
name="sc"  # Default job name

# Parse command-line arguments
while getopts ":f:t:g:n:" opt; do
    case ${opt} in
        f )
            script_path=$OPTARG
            ;;
        t )
            time_limit=$OPTARG
            ;;
        g )
            gpu_count=$OPTARG
            ;;
        n )
            name=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$script_path" ]; then
    echo "No script provided"
    usage
fi

# Extract the script name and replace slashes with hyphens
job_name=$(echo "$script_path" | sed 's/\//-/g' | sed 's/\.sh$//')

# Create a temporary SLURM script
tmp_script=$(mktemp jobscript.XXXXXX.sh)

cat <<EOT > $tmp_script
#!/bin/bash -l

#SBATCH -J $name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=$time_limit
#SBATCH --account=plgmeetween2026-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:$gpu_count
#SBATCH --output="z_outputs/${job_name}-%j.out-%N"
#SBATCH --error="z_outputs/${job_name}-%j.err-%N"

module load Miniconda3/23.3.1-0
eval "\$(conda shell.bash hook)"
module load CUDA/12.4

source ~/.bashrc
module load CUDA/12.4
conda activate qwen3
module load CUDA/12.4
module load  GCCcore/13.2.0
module load  FFmpeg/6.0
srun bash "$script_path"
EOT

# Submit the temporary script to SLURM
sbatch $tmp_script

# Clean up the temporary script
rm $tmp_script
