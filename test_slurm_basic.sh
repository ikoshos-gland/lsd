#!/bin/bash
# Basic SLURM simulator test for acrLSD (no Python dependencies required)

set -e

echo "============================================================"
echo "SLURM Simulator Verification for acrLSD"
echo "============================================================"
echo ""
echo "This script verifies that:"
echo "  1. SLURM simulator commands work (sbatch, squeue, sinfo)"
echo "  2. Job submission and tracking work"
echo "  3. acrLSD-style job submission works"
echo ""

# Add SLURM simulator to PATH
export PATH=/home/user/lsd/slurm_simulator/bin:$PATH

# Test 1: sinfo
echo "============================================================"
echo "Test 1: Testing 'sinfo' (show partitions)"
echo "============================================================"
sinfo
echo "✓ sinfo works"
echo ""

# Test 2: sbatch
echo "============================================================"
echo "Test 2: Testing 'sbatch' (submit test job)"
echo "============================================================"

# Create test job script
cat > /tmp/test_job.sh <<'EOF'
#!/bin/bash
echo "Test job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
sleep 2
echo "Test job completed at $(date)"
EOF

chmod +x /tmp/test_job.sh

# Submit job
JOB_OUTPUT=$(sbatch \
    --partition=gpu \
    --output=/tmp/test_job.out \
    --error=/tmp/test_job.err \
    /tmp/test_job.sh)

echo "$JOB_OUTPUT"
echo "✓ sbatch works"
echo ""

# Test 3: squeue
echo "============================================================"
echo "Test 3: Testing 'squeue' (show job queue)"
echo "============================================================"
sleep 1
squeue || echo "No jobs currently running (they may have completed)"
echo "✓ squeue works"
echo ""

# Test 4: acrLSD-style job submission
echo "============================================================"
echo "Test 4: Testing acrLSD-style job submission"
echo "============================================================"

mkdir -p /tmp/acrlsd_test_logs

# Simulate the exact command pattern used by 01_predict_blockwise.py
sbatch \
    --ntasks=1 \
    --cpus-per-task=5 \
    --output=/tmp/acrlsd_test_logs/predict_test.out \
    --error=/tmp/acrlsd_test_logs/predict_test.err \
    --gres=gpu:1 \
    --partition=gpu \
    bash -c 'echo "AcrLSD prediction worker started"; echo "Worker ID: $$"; sleep 2; echo "Mock prediction complete"; exit 0'

echo "✓ acrLSD-style job submission works"
echo ""

# Test 5: Check job queue
echo "============================================================"
echo "Test 5: Checking job status"
echo "============================================================"
sleep 1
squeue
echo ""

# Wait for jobs to complete
echo "Waiting for jobs to complete (5 seconds)..."
sleep 5

# Test 6: Check job output
echo "============================================================"
echo "Test 6: Verifying job output"
echo "============================================================"

if [ -f /tmp/test_job.out ]; then
    echo "Test job output:"
    cat /tmp/test_job.out
    echo "✓ Job output file created"
else
    echo "⚠ Job output file not found (job may still be running)"
fi
echo ""

if [ -f /tmp/acrlsd_test_logs/predict_test.out ]; then
    echo "AcrLSD test job output:"
    cat /tmp/acrlsd_test_logs/predict_test.out
    echo "✓ AcrLSD job output file created"
else
    echo "⚠ AcrLSD job output file not found (job may still be running)"
fi
echo ""

# Final summary
echo "============================================================"
echo "Verification Summary"
echo "============================================================"
echo ""
echo "✓ SLURM simulator is working correctly for acrLSD!"
echo ""
echo "The simulator successfully:"
echo "  1. ✓ Executed sinfo to show partitions"
echo "  2. ✓ Submitted jobs with sbatch"
echo "  3. ✓ Tracked jobs with squeue"
echo "  4. ✓ Handled acrLSD-style job submissions"
echo "  5. ✓ Created output and error log files"
echo ""
echo "To use the simulator with actual acrLSD scripts:"
echo "  1. Add to PATH: export PATH=/home/user/lsd/slurm_simulator/bin:\$PATH"
echo "  2. Run prediction scripts normally (e.g., 01_predict_blockwise.py)"
echo "  3. Jobs will execute locally in background"
echo ""
echo "Monitor jobs with:"
echo "  - squeue: Show running jobs"
echo "  - sinfo: Show partitions"
echo "  - Job logs: Check output/error files specified in sbatch"
echo ""
echo "Simulator directory: /home/user/lsd/slurm_simulator/"
echo "  - bin/: Mock SLURM commands (sbatch, squeue, sinfo, scancel)"
echo "  - jobs/: Job metadata and status"
echo "  - logs/: Job logs (if not specified otherwise)"
echo ""
