# Distributed Training and Inference in LSD

This document provides a comprehensive overview of distributed training and inference approaches used in the Local Shape Descriptors (LSD) project for large-scale neuron segmentation.

## Table of Contents

1. [Introduction](#introduction)
2. [Distributed Inference with Daisy](#distributed-inference-with-daisy)
3. [Distributed Training Concepts](#distributed-training-concepts)
4. [Framework-Specific Training](#framework-specific-training)
5. [Implementation in LSD Project](#implementation-in-lsd-project)
6. [Best Practices & Recommendations](#best-practices--recommendations)

---

## Introduction

### Why Distributed Processing?

Neuroscience datasets, particularly electron microscopy (EM) volumes, can range from terabytes to petabytes in size. Processing such volumes requires:

- **Distributed Inference**: Splitting large volumes into blocks and processing them in parallel across multiple workers
- **Distributed Training**: Leveraging multiple GPUs to speed up training or enable training of larger models

The LSD project uses two primary frameworks for distributed computing:
- **Daisy**: Block-wise task scheduling for distributed inference across clusters
- **Gunpowder**: Pipeline framework with built-in parallelization for data augmentation

---

## Distributed Inference with Daisy

[Daisy](https://github.com/funkelab/daisy) is a block-wise task scheduling framework specifically designed for processing large nD volumes in neuroscience applications. It was developed by researchers at HHMI Janelia and Harvard.

### Architecture Overview

Daisy uses a **coordinator-worker** architecture:

1. **Coordinator**: Divides the total ROI into blocks and distributes them to workers
2. **Workers**: Process individual blocks and report completion
3. **MongoDB**: Tracks completed blocks and stores intermediate results (RAG nodes/edges)
4. **Cluster Scheduler**: Launches workers via SLURM (sbatch) for distributed processing

### Three-Stage Segmentation Pipeline

The standard distributed segmentation workflow consists of three stages:

#### 1. Predict (01_predict_blockwise.py:236)

```python
daisy.run_blockwise(
    total_roi,              # Full volume to process
    read_roi,               # Block size + context for input
    write_roi,              # Block size for output
    process_function=...,   # Function to launch workers
    check_function=...,     # Check if block completed
    num_workers=N,          # Number of parallel workers
    read_write_conflict=False,
    fit='overhang'
)
```

**What it does**: Runs neural network inference to generate affinities/LSDs from raw data

**Key features**:
- Each worker loads a trained model checkpoint
- Workers process blocks with context (overlap) to avoid boundary artifacts
- Uses `DaisyRequestBlocks` Gunpowder node to coordinate with Daisy scheduler
- Progress tracked in MongoDB `blocks_predicted` collection

**Worker structure** (vanilla/predict.py:110):
```python
pipeline += DaisyRequestBlocks(
    chunk_request,
    roi_map={'raw': 'read_roi', 'affs': 'write_roi'},
    num_workers=worker_config['num_cache_workers'],
    block_done_callback=lambda b, s, d: block_done_callback(...)
)
```

#### 2. Extract Fragments (02_extract_fragments_blockwise.py:182)

**What it does**: Performs watershed segmentation to create supervoxels (fragments) from affinities

**Key features**:
- Creates supervoxels with unique IDs across blocks
- Writes RAG nodes (fragment centers) to MongoDB
- Handles boundary consistency through context overlap
- Stores fragments in zarr/n5 format

**Worker coordination** (workers/extract_fragments_worker.py:74):
```python
client = daisy.Client()
while True:
    block = client.acquire_block()  # Request block from coordinator
    if block is None:
        break

    # Process block...
    lsd.watershed_in_block(affs, block, context, rag_provider, fragments, ...)

    client.release_block(block, ret=0)  # Report completion
```

#### 3. Agglomerate (03_agglomerate_blockwise.py:125)

**What it does**: Computes edge weights in the Region Adjacency Graph (RAG) for hierarchical merging

**Key features**:
- Reads fragment nodes from MongoDB RAG
- Computes merge scores between adjacent fragments
- Writes edge weights to MongoDB
- Enables threshold-based segmentation (04_find_segments.py)

### Block Management

**Read vs Write ROI**:
```python
# Example: net input=196^3, output=84^3, voxel_size=8nm
context = (net_input_size - net_output_size) / 2  # 56 voxels = 448nm

read_roi = Roi((0,0,0), block_size).grow(context, context)   # Block + context
write_roi = Roi((0,0,0), block_size)                         # Block only
```

**Why context?**
- Neural networks have a receptive field - need surrounding pixels for accurate prediction
- Watershed needs context to properly connect fragments at boundaries
- Context overlap ensures consistency across block boundaries

### Cluster Integration

Workers are launched via cluster schedulers (lsd/tutorial/scripts/01_predict_blockwise.py:330):

```python
command = [
    'sbatch',                                              # SLURM scheduler
    '--ntasks=1',                                          # Number of tasks
    '--cpus-per-task=' + str(worker_config['num_cpus']),  # CPUs per worker
    '--output=' + f'{log_out}',                           # Output log
    '--error=' + f'{log_err}',                            # Error log
    '--gres=gpu:1',                                        # Request 1 GPU
    '--partition=' + worker_config['queue']                # Partition name (gpu, etc)
]

# Optional: run in Singularity container
if singularity_image is not None:
    command += ['singularity exec', '--nv', singularity_image]

command += [f'python -u {predict_script} {config_file}']
```

**Note**: SLURM uses "partitions" instead of "queues". Common partition names: `gpu`, `cpu`, `high-priority`

### Parallel Processing Functions

The `lsd.post` module provides high-level parallel processing functions:

**parallel_fragments.py** (lsd/post/parallel_fragments.py:125):
```python
lsd.parallel_watershed(
    affs,                    # Affinity predictions (daisy.Array)
    rag_provider,            # MongoDB or file-based RAG storage
    block_size,              # Size of blocks to process
    context,                 # Context for boundaries
    fragments_out,           # Output array for fragments
    num_workers,             # Number of parallel workers
    fragments_in_xy=False,   # 2D vs 3D watershed
    epsilon_agglomerate=0.0  # Optional initial agglomeration
)
```

**parallel_lsd_agglomerate.py** (lsd/post/parallel_lsd_agglomerate.py:66):
```python
lsd.parallel_lsd_agglomerate(
    lsds,              # Predicted LSDs (daisy.Array)
    fragments,         # Supervoxels from watershed
    rag_provider,      # RAG storage backend
    lsd_extractor,     # LSD computation object
    block_size,        # Block size
    context,           # Context size
    num_workers        # Parallel workers
)
```

These functions internally use `daisy.run_blockwise()` and handle the block processing logic directly (no cluster scheduler required).

### MongoDB Storage

Daisy uses MongoDB for:

1. **Block tracking**: Record completed blocks to enable resumption after failures
   ```python
   blocks_predicted.insert({'block_id': block.block_id, 'duration': ...})
   ```

2. **RAG storage**: Store graph nodes (fragments) and edges (adjacencies)
   ```python
   rag_provider = daisy.persistence.MongoDbGraphProvider(
       db_name,
       host=db_host,
       position_attribute=['center_z', 'center_y', 'center_x']
   )
   ```

3. **Alternative**: Use `FileGraphProvider` for local processing without MongoDB

---

## Distributed Training Concepts

### Data Parallelism

**Concept**: Split data across workers, replicate model on each worker

**How it works**:
1. Each GPU maintains a full copy of the model
2. Each GPU processes a different batch of data
3. Gradients are synchronized across GPUs (all-reduce operation)
4. Model parameters updated with averaged gradients

**Pros**:
- Simple to implement
- Works with most models
- Near-linear speedup with good network

**Cons**:
- Each GPU needs full model in memory
- Communication overhead for large models
- Limited by single-GPU model capacity

**Use when**: Model fits on single GPU, want to speed up training with larger effective batch size

### Model Parallelism

**Concept**: Split model across workers, same data on each worker

**Types**:

1. **Pipeline Parallelism**: Split model into sequential stages across GPUs
   - Example: Layers 1-10 on GPU0, layers 11-20 on GPU1, etc.
   - Requires micro-batching to keep all GPUs busy

2. **Tensor Parallelism**: Split individual layers across GPUs
   - Example: Large matrix multiplications computed across multiple GPUs
   - Lower latency than pipeline parallelism
   - More complex communication patterns

**Pros**:
- Enables training models larger than single GPU memory
- Reduces memory per GPU

**Cons**:
- More complex implementation
- Communication overhead between model partitions
- May not fully utilize all GPUs (pipeline bubbles)

**Use when**: Model doesn't fit on single GPU, even with gradient checkpointing

### Hybrid Approaches

Large-scale models (GPT-3, T5) use combinations:
- Data parallelism across nodes
- Model parallelism within nodes
- Example: 4-way tensor parallel + 8-way data parallel = 32 GPUs

---

## Framework-Specific Training

### PyTorch Distributed Training

#### DistributedDataParallel (DDP)

**Standard data parallelism** for multi-GPU training:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')  # NCCL for GPU, Gloo for CPU

# Create model and move to GPU
local_rank = int(os.environ['LOCAL_RANK'])
model = UNet(...).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training loop - each process gets different data
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()  # Gradients automatically synchronized
    optimizer.step()
```

**Key features**:
- Each process owns 1 GPU (recommended: 1 process per GPU)
- Automatic gradient synchronization via all-reduce
- Efficient ring-reduce algorithm minimizes communication
- Works with standard PyTorch data loaders (use DistributedSampler)

**Launch**:
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node (2 nodes, 4 GPUs each)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=node0 train.py
```

#### Fully Sharded Data Parallel (FSDP / FSDP2)

**Memory-efficient training** by sharding model parameters:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = UNet(...)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Shard params, grads, optimizer
    mixed_precision=...,
    cpu_offload=...  # Offload to CPU for even larger models
)
```

**How it works**:
1. Model parameters sharded across GPUs (each GPU stores 1/N of parameters)
2. During forward pass: all-gather needed parameters
3. During backward pass: all-gather for gradient computation
4. Optimizer step: only update local shard

**FSDP2 improvements (2025)**:
- Better memory management (no recordStream overhead)
- Mix frozen/non-frozen parameters efficiently
- DTensor representation for easier manipulation
- Simpler meta-device initialization

**Use when**:
- Model doesn't fit on single GPU with DDP
- Want to train larger models or use larger batch sizes
- Acceptable to trade some speed for memory efficiency

**Performance**:
- DDP: Faster, but requires full model per GPU
- FSDP: Slower (more communication), but 1/N memory per GPU

### TensorFlow Distributed Training

#### MirroredStrategy

**Single-machine multi-GPU training**:

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()  # Auto-detects all GPUs

with strategy.scope():
    model = create_model()  # Build model within strategy scope
    model.compile(optimizer='adam', loss='mse')

# Training automatically distributed
model.fit(dataset, epochs=10)
```

**How it works**:
- Creates replica of model on each GPU
- Uses all-reduce to combine gradients (NCCL or hierarchical all-reduce)
- Synchronous updates across all replicas

#### MultiWorkerMirroredStrategy

**Multi-machine distributed training**:

```python
import tensorflow as tf
import json

# Configure cluster
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.NCCL
    )
)

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='mse')

# Each worker processes different data
model.fit(dataset, epochs=10)
```

**Launch** (configure via TF_CONFIG environment variable):
```python
# Worker 0
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['host1:12345', 'host2:12345']
    },
    'task': {'type': 'worker', 'index': 0}
})

# Worker 1
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['host1:12345', 'host2:12345']
    },
    'task': {'type': 'worker', 'index': 1}
})
```

### Gunpowder Pipeline Parallelism

Gunpowder provides **data loading and augmentation parallelism** through the `PreCache` node:

```python
from gunpowder import *

pipeline = (
    # Data sources and augmentation
    ZarrSource(...) +
    RandomLocation() +
    ElasticAugment(...) +
    SimpleAugment() +
    IntensityAugment(...) +

    # Parallel data preprocessing
    PreCache(
        cache_size=40,      # Number of batches to pre-compute
        num_workers=10      # Parallel augmentation workers
    ) +

    # Training (single GPU)
    Train(model, loss, optimizer, ...)
)
```

**How PreCache works**:
1. Spawns `num_workers` processes using Python multiprocessing
2. Each worker independently augments data
3. Pre-computed batches stored in queue (size=cache_size)
4. Training pulls from queue, never waits for augmentation

**Performance impact**:
- Without PreCache: Training waits for each batch to be augmented
- With PreCache: Augmentation parallelized, training GPU always busy
- Rule of thumb: `num_workers` = number of CPU cores / 2

**Multiprocessing considerations**:
- Uses 'fork' spawn method by default
- Can conflict with CUDA (use 'spawn' for multi-GPU)
- Each worker loads full dataset in memory

---

## Implementation in LSD Project

### Current Training Setup

**Single-GPU training with data parallelism** (via PreCache):

```python
# lsd/tutorial/example_nets/fib25/vanilla/train_pytorch.py:179
train_pipeline += PreCache(
    cache_size=40,
    num_workers=10
)

train_pipeline += Train(
    model=model,
    loss=loss,
    optimizer=optimizer,
    ...
)
```

**Training characteristics**:
- Single GPU training loop
- Data parallelism through PreCache (10 workers)
- No multi-GPU model parallelism currently implemented
- TensorFlow version also single-GPU

### Distributed Inference Setup

**Block-wise parallel inference** with cluster scheduling:

```python
# lsd/tutorial/scripts/01_predict_blockwise.py
predict_blockwise(
    base_dir='.',
    experiment='fib25',
    setup='setup01',
    iteration=100000,
    raw_file='data.zarr',
    raw_dataset='volumes/raw',
    out_base='predictions',
    file_name='predictions.zarr',
    num_workers=20,        # 20 parallel GPU workers
    db_host='localhost',
    db_name='fib25_predictions',
    queue='gpu'  # SLURM partition name
)
```

**Each worker** (vanilla/predict.py):
1. Loads trained model checkpoint
2. Receives blocks from Daisy coordinator via DaisyRequestBlocks
3. Runs inference on block + context
4. Writes output to shared zarr/n5 file
5. Reports completion to MongoDB

### Extending to Multi-GPU Training

**Option 1: PyTorch DDP**

Modify training script to use DDP:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# Wrap model
model = UNet(...).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler for data loading
# Gunpowder doesn't have built-in DistributedSampler, so you'd need to:
# - Run separate Gunpowder pipeline per process
# - Use different random seeds per process
# - Or implement custom node for distributed data sharding
```

**Option 2: FSDP for Large Models**

If model doesn't fit on single GPU:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = UNet(...)
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

**Option 3: TensorFlow MultiWorkerMirroredStrategy**

For TensorFlow training scripts:

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    # Build model and training within strategy scope
    # Gunpowder pipeline runs outside strategy (data loading)
```

**Challenges**:
- Gunpowder not designed for distributed training
- Would need to run separate Gunpowder pipeline per worker
- Or re-implement data loading with PyTorch DistributedSampler

### Configuration Considerations

**Block size selection**:
- Trade-off between parallelism and context overhead
- Smaller blocks: More parallelism, but more context redundancy
- Larger blocks: Less overhead, but fewer parallel tasks
- Rule of thumb: Block size ≥ 2× network input size

**Number of workers**:
- Limited by: GPU availability, network bandwidth, storage I/O
- Start with num_workers = num_GPUs
- Monitor GPU utilization, increase if GPUs idle

**Context size**:
- Prediction: context = (input_size - output_size) / 2
- Watershed: additional context helps boundary consistency (typically 40-80nm)
- Agglomeration: similar to watershed context

**Memory considerations**:
- Each prediction worker loads full model (~few GB)
- Workers share zarr/n5 files (read: multiple, write: exclusive per block)
- MongoDB RAM usage grows with number of fragments (RAG nodes/edges)

### Deployment on Azure CycleCloud with SLURM

The LSD distributed inference pipeline can be deployed on **Azure CycleCloud** using SLURM as the job scheduler. Azure CycleCloud provides auto-scaling HPC clusters that are ideal for large-scale neuroscience workloads.

#### Why Azure CycleCloud + SLURM?

**Benefits**:
- **Auto-scaling**: VMs automatically created when jobs queued, shut down when idle
- **GPU variety**: Choose from NC, ND, NCv3, NDv2 series (V100, A100, H100)
- **Cost optimization**: Mix spot (low-priority) and on-demand instances
- **No infrastructure management**: No physical cluster to maintain
- **Native Azure integration**: Access to Azure NetApp Files, Blob Storage, Cosmos DB

**Cost savings**:
- Spot instances: ~70% cheaper than on-demand (e.g., NC6s_v3: $0.27/hr vs $0.90/hr)
- Pay only for actual compute time
- Auto-shutdown after idle timeout eliminates waste

#### Cluster Setup

**1. Create SLURM cluster in Azure CycleCloud**:

```bash
# Install Azure CycleCloud CLI and import cluster template
git clone https://github.com/Azure/cyclecloud-slurm
cd cyclecloud-slurm
cyclecloud import_cluster lsd-cluster -f templates/slurm.txt
```

**2. Configure GPU and CPU partitions**:

```ini
# Cluster configuration (modify in Azure Portal or cluster template)
[cluster lsd-cluster]
FormLayout = selectionpanel

[[node defaults]]
    Credentials = $Credentials
    Region = eastus

[[node scheduler]]
    MachineType = Standard_D4s_v3  # Scheduler node (4 cores, 16GB RAM)
    ImageName = cycle.image.ubuntu20

[[nodearray gpu]]
    # GPU partition for prediction
    MachineType = Standard_NC6s_v3      # 1x V100 GPU, 6 cores, 112GB RAM
    ImageName = microsoft-dsvm:ubuntu-hpc:2004:latest  # Ubuntu 20.04 with CUDA

    # Auto-scaling settings
    Interruptible = false               # On-demand instances
    MaxCoreCount = 240                  # Max 40 VMs (40 × 6 cores)

    [[[configuration]]]
    slurm.autoscale = true              # Enable auto-scaling
    slurm.default_partition = false

    [[[cluster-init cyclecloud/slurm:execute]]]

[[nodearray gpulowprio]]
    # Low-priority GPU partition (spot instances)
    MachineType = Standard_NC6s_v3
    ImageName = microsoft-dsvm:ubuntu-hpc:2004:latest

    Interruptible = true                # Spot instances (can be preempted)
    MaxCoreCount = 120                  # Max 20 VMs

    [[[configuration]]]
    slurm.autoscale = true

[[nodearray cpu]]
    # CPU partition for watershed/agglomeration
    MachineType = Standard_F16s_v2      # 16 cores, 32GB RAM
    ImageName = cycle.image.ubuntu20

    MaxCoreCount = 320                  # Max 20 VMs (20 × 16 cores)

    [[[configuration]]]
    slurm.autoscale = true
```

**3. Configure SLURM partitions** (on scheduler node: `/sched/slurm/slurm.conf.d/`):

```bash
# GPU partition for neural network inference
PartitionName=gpu Nodes=ALL Default=NO MaxTime=INFINITE State=UP \
    DefMemPerCPU=16000 OverSubscribe=NO Priority=100

# Low-priority GPU partition
PartitionName=gpulowprio Nodes=ALL Default=NO MaxTime=24:00:00 State=UP \
    DefMemPerCPU=16000 OverSubscribe=NO Priority=50

# CPU partition for post-processing
PartitionName=cpu Nodes=ALL Default=YES MaxTime=INFINITE State=UP \
    DefMemPerCPU=2000 OverSubscribe=NO Priority=75
```

**4. Set up shared storage**:

Choose one of these options:

**Option A: Azure NetApp Files** (recommended for performance):
```bash
# Create NetApp account and volume via Azure Portal
# Mount on all nodes (add to cluster-init script)
sudo mkdir -p /mnt/shared
sudo mount -t nfs <netapp-ip>:/volume /mnt/shared
```

**Option B: Azure Blob Storage with NFS 3.0**:
```bash
# Enable NFS 3.0 on storage account
# Mount on all nodes
sudo mkdir -p /mnt/shared
sudo mount -o sec=sys,vers=3,proto=tcp <account>.blob.core.windows.net:/<account>/<container> /mnt/shared
```

**Directory structure**:
```
/mnt/shared/
├── data/              # Input datasets
├── models/            # Trained model checkpoints
├── predictions/       # Inference outputs
└── results/           # Final segmentations
```

**5. Set up MongoDB**:

**Option A: MongoDB on scheduler node** (simple, cheap):
```bash
# On scheduler node
sudo apt-get install -y mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb

# Configure to accept remote connections
sudo nano /etc/mongodb.conf
# Set: bind_ip = 0.0.0.0

# In Python scripts, use:
db_host = '<scheduler-node-ip>'  # e.g., 10.0.0.4
```

**Option B: Azure Cosmos DB for MongoDB** (managed, scalable):
```python
# More expensive but fully managed
db_host = '<cosmosdb-account>.mongo.cosmos.azure.com:10255'
connection_string = 'mongodb://...'  # With authentication
```

#### Running the Pipeline on Azure

**1. Install LSD environment on all nodes** (cluster-init script):

```bash
#!/bin/bash
# This runs on each node when it starts

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3

# Create LSD environment
/opt/miniconda3/bin/conda create -n lsd python=3.10 -y
/opt/miniconda3/bin/conda install -n lsd -c conda-forge \
    lsds daisy pymongo zarr tensorstore -y

# Install PyTorch with CUDA support (GPU nodes only)
if [[ -d /usr/local/cuda ]]; then
    /opt/miniconda3/bin/conda install -n lsd \
        pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

# Mount shared storage (if not in cluster config)
sudo mkdir -p /mnt/shared
sudo mount -t nfs <netapp-ip>:/volume /mnt/shared

# Add conda to PATH
echo 'export PATH=/opt/miniconda3/envs/lsd/bin:$PATH' >> /etc/profile.d/conda.sh
```

**2. Submit pipeline jobs**:

```bash
# SSH to scheduler node
ssh azureuser@<scheduler-ip>

# Activate environment
source /opt/miniconda3/bin/activate lsd

# Navigate to shared directory
cd /mnt/shared/lsd

# Step 1: Prediction (GPU partition)
python scripts/01_predict_blockwise.py \
    --base-dir /mnt/shared \
    --experiment fib25 \
    --setup setup01 \
    --iteration 100000 \
    --raw-file /mnt/shared/data/sample.zarr \
    --raw-dataset volumes/raw \
    --out-base /mnt/shared/predictions \
    --file-name predictions.zarr \
    --num-workers 20 \
    --db-host 10.0.0.4 \
    --db-name lsd_predictions \
    --queue gpu

# Step 2: Watershed (CPU partition)
python scripts/02_extract_fragments_blockwise.py \
    --experiment fib25 \
    --setup setup01 \
    --iteration 100000 \
    --affs-file /mnt/shared/predictions/setup01/100000/predictions.zarr \
    --affs-dataset volumes/affs \
    --fragments-file /mnt/shared/predictions/setup01/100000/fragments.zarr \
    --fragments-dataset volumes/fragments \
    --num-workers 10 \
    --db-host 10.0.0.4 \
    --db-name lsd_predictions \
    --queue cpu

# Step 3: Agglomerate (CPU partition)
python scripts/03_agglomerate_blockwise.py \
    --experiment fib25 \
    --setup setup01 \
    --iteration 100000 \
    --affs-file /mnt/shared/predictions/setup01/100000/predictions.zarr \
    --affs-dataset volumes/affs \
    --fragments-file /mnt/shared/predictions/setup01/100000/fragments.zarr \
    --fragments-dataset volumes/fragments \
    --num-workers 10 \
    --db-host 10.0.0.4 \
    --db-name lsd_predictions \
    --queue cpu
```

**3. Monitor jobs**:

```bash
# View job queue
squeue

# View partition status
sinfo

# View specific job details
scontrol show job <job-id>

# Cancel job
scancel <job-id>

# View node status
sinfo -N -l
```

#### Cost Optimization Strategies

**1. Mix spot and on-demand instances**:
```python
# Use spot instances for non-critical stages
queue = 'gpulowprio'  # For testing or low-priority work
queue = 'gpu'         # For production inference
```

**2. Configure auto-scale idle timeout**:
```ini
# In cluster config
slurm.autoscale = true
slurm.idle_timeout = 300  # Shut down after 5 minutes idle
```

**3. Use appropriate VM sizes**:
- **Prediction**: NC6s_v3 (1 GPU) or NC12s_v3 (2 GPUs)
- **Watershed/Agglomerate**: F16s_v2 (CPU-optimized, 16 cores)
- **Testing**: B-series (burstable, cheap)

**4. Cost estimates** (eastus region, approximate):

| Stage | VM Type | Count | Hours | Cost (on-demand) | Cost (spot) |
|-------|---------|-------|-------|------------------|-------------|
| Prediction | NC6s_v3 | 20 | 4 | $72 | $22 |
| Watershed | F16s_v2 | 10 | 2 | $14 | $4 |
| Agglomerate | F16s_v2 | 10 | 2 | $14 | $4 |
| **Total** | | | | **$100** | **$30** |

**Tip**: Use spot instances for development/testing, on-demand for production.

#### Advantages vs On-Premises Cluster

| Feature | Azure CycleCloud | On-Premises |
|---------|------------------|-------------|
| Capital cost | $0 (pay-as-you-go) | $100K-1M+ |
| Scalability | Unlimited (Azure quota) | Fixed hardware |
| Maintenance | None (managed) | IT staff required |
| GPU upgrades | Instant (change VM type) | Hardware refresh cycle |
| Idle cost | $0 (auto-shutdown) | Full power/cooling |
| Geographic distribution | Multi-region | Single location |

#### Troubleshooting

**Issue: Nodes not auto-scaling**

```bash
# Check autoscale status
sudo azslurm scale

# View autoscale logs
tail -f /opt/azurehpc/slurm/logs/autoscale.log

# Manually trigger scale operation
sudo azslurm scale --partition gpu
```

**Issue: GPU not detected**

```bash
# Verify CUDA installation
nvidia-smi

# Check SLURM GPU configuration
scontrol show node <node-name> | grep Gres

# Reconfigure GPUs
sudo /opt/azurehpc/slurm/scripts/configure_gres.sh
```

**Issue: Slow storage access**

```bash
# Test NetApp Files throughput
dd if=/dev/zero of=/mnt/shared/test bs=1M count=10000

# Consider Azure Premium Files or increase NetApp service level
# NetApp tiers: Standard (16 MiB/s/TiB), Premium (64 MiB/s/TiB), Ultra (128 MiB/s/TiB)
```

---

## Best Practices & Recommendations

### When to Use Data vs Model Parallelism

**Use Data Parallelism (DDP) when**:
- Model fits comfortably on single GPU
- Have multiple GPUs available
- Want to speed up training
- Want larger effective batch sizes

**Use Model Parallelism (FSDP) when**:
- Model doesn't fit on single GPU
- Willing to trade speed for memory
- Have high-bandwidth interconnect (NVLink, InfiniBand)

**Use Hybrid when**:
- Training very large models (billions of parameters)
- Have large GPU clusters
- Can afford complex setup

### Memory Optimization

**Techniques**:

1. **Gradient checkpointing**: Recompute activations instead of storing
   ```python
   from torch.utils.checkpoint import checkpoint

   def forward(self, x):
       x = checkpoint(self.layer1, x)  # Don't store activations
       x = checkpoint(self.layer2, x)
       return x
   ```

2. **Mixed precision training**: Use FP16 instead of FP32
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       loss = model(batch)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   ```

3. **Gradient accumulation**: Simulate larger batch size
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Activation offloading**: Move activations to CPU during forward pass
   - FSDP supports `cpu_offload=CPUOffload(offload_params=True)`

### Network and Storage Bottlenecks

**Network**:
- High-bandwidth interconnect essential for model parallelism
- Gradient synchronization scales with model size
- Use NCCL for GPU-GPU communication (optimized for NVIDIA)
- Monitor network utilization during training

**Storage**:
- Shared filesystems can bottleneck with many workers
- Use local SSDs for temporary data when possible
- Zarr/N5 with compression reduces I/O
- Consider chunking strategy (align with block size)

**MongoDB**:
- RAG storage can grow very large (billions of nodes/edges)
- Index properly (`block_id`, `node_id`)
- Consider sharding for very large datasets
- Alternative: `FileGraphProvider` for local processing

### Debugging Distributed Systems

**Common issues**:

1. **Deadlocks**: Workers waiting for blocks that don't exist
   - Check total_roi vs block_size alignment
   - Verify check_function logic

2. **Out of memory**: Worker exceeds GPU memory
   - Reduce block size or network input size
   - Enable mixed precision training
   - Use gradient checkpointing

3. **Slow training**: GPUs underutilized
   - Increase PreCache num_workers
   - Increase cache_size
   - Check data loading bottlenecks

4. **NaN losses**: Numerical instability in distributed training
   - Reduce learning rate
   - Use gradient clipping
   - Check for race conditions in custom nodes

**Debugging tools**:
- `nvidia-smi`: Monitor GPU utilization and memory
- `htop`: CPU utilization
- `iftop`: Network bandwidth
- PyTorch Profiler: Detailed performance analysis
- MongoDB logs: Check for slow queries

### Monitoring and Logging

**What to track**:
- Training: loss, learning rate, GPU utilization, throughput (samples/sec)
- Inference: blocks completed, failures, average duration per block
- System: GPU memory, network I/O, storage I/O

**Tools**:
- TensorBoard: Training metrics and profiling
- MLflow: Experiment tracking
- MongoDB: Block completion and timing
- Grafana + Prometheus: System monitoring

---

## Further Reading

**Daisy**:
- GitHub: https://github.com/funkelab/daisy
- Documentation: https://funkelab.github.io/daisy

**Gunpowder**:
- GitHub: https://github.com/funkelab/gunpowder
- Documentation: https://funkelab.github.io/gunpowder

**PyTorch Distributed**:
- DDP Tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- FSDP Tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

**TensorFlow Distributed**:
- Guide: https://www.tensorflow.org/guide/distributed_training

**General Distributed Training**:
- "Demystifying Parallel and Distributed Deep Learning" (arXiv:1802.09941)
- "Data-Parallel Distributed Training": https://siboehm.com/articles/22/data-parallel-training

---

## Summary

The LSD project demonstrates two complementary approaches to distributed computing:

1. **Distributed Inference**: Uses Daisy for block-wise parallel processing of large volumes
   - Coordinator-worker architecture
   - Cluster scheduler integration (SLURM)
   - MongoDB for progress tracking and RAG storage
   - Three-stage pipeline: predict → watershed → agglomerate

2. **Training Parallelism**: Currently uses Gunpowder PreCache for data augmentation parallelism
   - Single-GPU training in example scripts
   - Can be extended to multi-GPU with DDP or FSDP
   - TensorFlow scripts can use MirroredStrategy or MultiWorkerMirroredStrategy

For most users, the **distributed inference** setup is more critical, as it enables processing TB-scale volumes that would be infeasible on a single machine. The training scripts can be adapted to multi-GPU setups when training time becomes a bottleneck or models grow beyond single-GPU capacity.
