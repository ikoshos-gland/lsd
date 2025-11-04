# SLURM Simulator for acrLSD Testing

## Overview

This is a lightweight SLURM simulator designed to test the acrLSD prediction pipeline without requiring an actual SLURM cluster. It implements the core SLURM commands used by the LSD blockwise processing scripts:

- `sbatch`: Submit batch jobs
- `squeue`: Show job queue status
- `sinfo`: Show partition information
- `scancel`: Cancel jobs

## Installation

The simulator is already installed at: `/home/user/lsd/slurm_simulator/`

### Directory Structure

```
slurm_simulator/
├── bin/
│   ├── sbatch     # Job submission command
│   ├── squeue     # Job queue status
│   ├── sinfo      # Partition information
│   └── scancel    # Cancel jobs
├── jobs/          # Job metadata (JSON files)
└── logs/          # Default log directory
```

## Usage

### 1. Add Simulator to PATH

```bash
export PATH=/home/user/lsd/slurm_simulator/bin:$PATH
```

### 2. Verify Installation

```bash
# Show available partitions
sinfo

# Check job queue (should be empty initially)
squeue
```

### 3. Test Basic Job Submission

```bash
# Create test script
echo '#!/bin/bash
echo "Hello from SLURM simulator"
sleep 2
echo "Job complete"' > /tmp/test.sh

chmod +x /tmp/test.sh

# Submit job
sbatch --partition=gpu --output=/tmp/test.out --error=/tmp/test.err /tmp/test.sh

# Check status
squeue

# Wait and check output
sleep 3
cat /tmp/test.out
```

## Integration with acrLSD

The simulator is fully compatible with the acrLSD prediction pipeline scripts:

### Example: Running 01_predict_blockwise.py

```python
import os

# Add simulator to PATH
os.environ['PATH'] = '/home/user/lsd/slurm_simulator/bin:' + os.environ.get('PATH', '')

# Import and run blockwise prediction
from lsd.tutorial.scripts.01_predict_blockwise import predict_blockwise

predict_blockwise(
    base_dir='/path/to/lsd',
    experiment='fib25',
    setup='acrlsd',
    iteration=400000,
    raw_file='/path/to/raw.zarr',
    raw_dataset='volumes/raw',
    out_base='/path/to/predictions',
    file_name='predictions.zarr',
    num_workers=4,                    # Number of parallel workers
    db_host='localhost',              # MongoDB host
    db_name='lsd_predictions',
    queue='gpu',                      # SLURM partition
    auto_file=None,
    auto_dataset=None,
    singularity_image=None
)
```

### How It Works

1. **Job Submission**: When `01_predict_blockwise.py` calls `sbatch`, the simulator:
   - Generates a unique job ID
   - Creates job metadata in `jobs/job_<ID>.json`
   - Executes the command in background
   - Returns immediately (like real SLURM)

2. **Job Execution**: Each job:
   - Runs asynchronously in a background process
   - Writes stdout to `--output` file
   - Writes stderr to `--error` file
   - Updates job status (PENDING → RUNNING → COMPLETED/FAILED)

3. **Job Monitoring**: The `squeue` command:
   - Reads job metadata from `jobs/` directory
   - Shows currently running jobs
   - Hides completed/failed jobs

## Supported SLURM Options

### sbatch Options

The simulator supports these common sbatch options used by acrLSD:

```bash
sbatch \
    --ntasks=1 \                      # Number of tasks (recorded but not enforced)
    --cpus-per-task=5 \               # CPUs per task (recorded but not enforced)
    --output=/path/to/output.out \    # Stdout file
    --error=/path/to/error.err \      # Stderr file
    --gres=gpu:1 \                    # GPU resources (recorded but not enforced)
    --partition=gpu \                 # Partition name (recorded)
    /path/to/script.sh                # Script to execute
```

### Partitions

The simulator provides two partitions (matching common cluster setups):

- `gpu`: For GPU-based prediction jobs
- `cpu`: For CPU-based post-processing (watershed, agglomeration)

## Limitations

This is a **testing simulator**, not a full SLURM implementation:

1. **No Resource Limits**: Jobs always run (no quota enforcement)
2. **No Scheduling**: Jobs execute immediately in background
3. **Single Node Only**: All jobs run on localhost
4. **No Job Arrays**: Each job must be submitted individually
5. **No MPI Support**: Each job runs in isolation
6. **Simplified Status**: Only tracks PENDING, RUNNING, COMPLETED, FAILED, CANCELLED

## Verification

Run the test script to verify the simulator is working:

```bash
bash /home/user/lsd/test_slurm_basic.sh
```

Expected output:
```
✓ SLURM simulator is working correctly for acrLSD!

The simulator successfully:
  1. ✓ Executed sinfo to show partitions
  2. ✓ Submitted jobs with sbatch
  3. ✓ Tracked jobs with squeue
  4. ✓ Handled acrLSD-style job submissions
  5. ✓ Created output and error log files
```

## Troubleshooting

### Jobs not appearing in squeue

- Jobs complete very quickly and are removed from the queue
- Check output/error files to see if job ran successfully

### Output files not created

- Ensure parent directories exist
- Check that paths are absolute, not relative
- Look for errors in the job metadata: `cat jobs/job_<ID>.json`

### Permission errors

- Ensure simulator scripts are executable: `chmod +x bin/*`
- Check write permissions for output directories

## Comparison with Real SLURM

| Feature | Real SLURM | Simulator |
|---------|-----------|-----------|
| Job submission | ✓ | ✓ |
| Job queuing | ✓ | ✓ (immediate execution) |
| Status tracking | ✓ | ✓ (basic) |
| Resource management | ✓ | ✗ (no enforcement) |
| Multi-node | ✓ | ✗ (single node only) |
| Job arrays | ✓ | ✗ |
| Dependencies | ✓ | ✗ |
| Priority scheduling | ✓ | ✗ |

## Use Cases

This simulator is ideal for:

- **Local Development**: Test acrLSD pipeline without cluster access
- **CI/CD Testing**: Automated testing of blockwise scripts
- **Debugging**: Understand job submission flow before cluster deployment
- **Education**: Learn SLURM workflow without cluster resources

## Next Steps

Once your scripts work with the simulator, deploying to a real SLURM cluster requires:

1. **Remove simulator from PATH**: Don't export the simulator PATH on the cluster
2. **Configure Real Partitions**: Update partition names to match your cluster
3. **Adjust Resource Requests**: Set appropriate `--cpus-per-task`, `--mem`, etc.
4. **Test with Small Data**: Verify on a small volume before scaling up

## Credits

Created for testing the acrLSD neuron segmentation pipeline from the Funkelab LSD package:
https://github.com/funkelab/lsd
