# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local Shape Descriptors (LSDs) is a research package for computing shape descriptors from instance segmentations to improve neuron boundary prediction and segmentation quality in EM (electron microscopy) image analysis. The package is used as an auxiliary training target for neural networks performing neuron segmentation.

## Installation & Setup

### Basic Installation
```bash
# Via conda
conda create -n lsd_env python=3
conda activate lsd_env
conda install lsds -c conda-forge

# Via pip
pip install lsds
```

### Development Installation
```bash
# Install in editable mode with full dependencies
make install-dev
# Or manually:
pip install -e .[full]
```

### Building from Source
The package includes Cython extensions that need compilation:
```bash
python setup.py install
```

## Testing

```bash
# Run all tests
make test
# Or manually:
python -m tests -v
```

Test files are located in `/tests/` and cover:
- `test_shape_descriptor.py` - Core LSD computation
- `test_components.py` - Component selection
- `test_pipeline.py` - Gunpowder pipeline integration
- `test_synthetic.py` - Synthetic data generation
- `test_strided.py` - Strided operations

## Architecture

### Core Module Structure

The codebase is organized into two main submodules:

**`lsd.train`** - Core LSD computation (minimal dependencies)
- `local_shape_descriptor.py` - Main LSD computation functions
  - `get_local_shape_descriptors()` - Standalone function for computing LSDs
  - `LsdExtractor` - Class-based extractor with caching for repeated computations
- `gp/add_local_shape_descriptor.py` - Gunpowder node for pipeline integration
  - `AddLocalShapeDescriptor` - BatchFilter for training pipelines

**`lsd.post`** - Post-processing operations (heavier dependencies)
- `fragments.py` - Watershed segmentation to create supervoxels
- `agglomerate.py` - Hierarchical agglomeration using RAGs
- `rag.py` - Region adjacency graph operations
- `persistence/` - MongoDB and SQLite storage backends

### LSD Components

LSDs consist of up to 10 statistical descriptors per voxel (3D) or 6 (2D):

**3D Components (0-9):**
- 0-2: Mean offsets (z, y, x)
- 3-5: Diagonal covariance (zz, yy, xx)
- 6-8: Orthogonal covariance/Pearson coefficients (zy, zx, yx)
- 9: Size/count

**2D Components (0-5):**
- 0-1: Mean offsets (y, x)
- 2-3: Diagonal covariance (yy, xx)
- 4: Pearson coefficient (yx)
- 5: Size/count

Use the `components` parameter to compute only specific descriptors (e.g., `"012"` for mean offsets only, `"6789"` for diagonal covariance + size).

### Key Computation Modes

- **Gaussian mode** (default): Uses Gaussian convolution with sigma as standard deviation. Context required = 3*sigma.
- **Sphere mode**: Uses uniform sphere with sigma as radius. Context required = sigma.

The `downsample` parameter enables faster computation on downsampled volumes.

### Training Pipeline Integration

LSDs integrate with [Gunpowder](https://funkelab.github.io/gunpowder/) for training:

1. Use `AddLocalShapeDescriptor` node in pipeline to compute LSDs on-the-fly from ground truth segmentations
2. Network predicts LSDs as auxiliary task alongside affinities
3. Common architectures: vanilla affinities, LSD-only, MTLSD (multi-task LSDs + affinities), autocontext (ACRLSD)

Example networks in `lsd/tutorial/example_nets/fib25/`:
- `vanilla/` - Baseline affinity prediction
- `lsd/` - LSD prediction only
- `mtlsd/` - Multi-task (LSDs + affinities)
- `acrlsd/` - **AutoContext** (two-stage approach):
  - Stage 1: Predict LSDs from raw data (pretrained)
  - Stage 2: Predict affinities from concatenated [raw + predicted LSDs]
  - Training uses `Predict` Gunpowder node to run LSD inference on-the-fly (train.py:156-166)
  - Network concatenates raw and LSD channels before U-Net (mknet.py:59)

### Segmentation Pipeline

The standard post-processing workflow (implemented with [daisy](https://github.com/funkelab/daisy) for parallelization):

1. **Predict** (`01_predict_blockwise.py`) - Network inference to generate affinities/LSDs
2. **Watershed** (`02_extract_fragments_blockwise.py`) - Create supervoxels from affinities, store RAG nodes
3. **Agglomerate** (`03_agglomerate_blockwise.py`) - Hierarchical agglomeration, create weighted RAG edges
4. **Find segments** (`04_find_segments.py`) - Cut graph at thresholds, generate lookup tables
5. **Extract segmentation** (`05_extract_segmentation_from_lut.py`) - Apply LUTs to relabel supervoxels

Scripts in `lsd/tutorial/scripts/` demonstrate this workflow but are designed for LSF/SLURM clusters and MongoDB. Adapt for local use by:
- Removing MongoDB dependencies (use `FileGraphProvider` instead of `MongoDbGraphProvider`)
- Adjusting block size/workers for available resources
- Using notebook examples for small-scale processing

### Framework Evolution

**TensorFlow (deprecated)**: Original paper networks used TensorFlow with a two-step process:
1. `mknet.py` - Create network placeholders and save config
2. `train.py` - Load config and train

For AcrLSD specifically, the implementation trains LSDs first, then uses the saved checkpoint during affinity training to predict LSDs on-the-fly via Gunpowder's `Predict` node. The predicted LSDs are concatenated with raw input before feeding into the affinity U-Net.

**PyTorch (recommended)**: Modern implementations combine network definition and training in single script. See `lsd/tutorial/example_nets/fib25/vanilla/train_pytorch.py` for example. For autocontext in PyTorch, you could eliminate the need to write LSDs to disk by implementing both networks in a single training script.

## Important Notes

- For lightweight LSD computation only, consider [lsd-lite](https://github.com/funkelab/lsd-lite) package
- Import LSDs with: `from lsd.train import local_shape_descriptor` or `from lsd.train import LsdExtractor`
- Gunpowder import: `from lsd.train.gp import AddLocalShapeDescriptor`
- Post-processing requires additional dependencies: `funlib.segment`, `funlib.evaluate`, networkx==2.2
- The package was developed for research; some scripts require customization for different environments
- Notebooks in `lsd/tutorial/notebooks/` provide the best starting point for learning the pipeline
- Parallel processing scripts assume cluster environment (LSF/SLURM) and need adaptation for local use

## Dependencies

Core dependencies (from setup.py):
- numpy, scipy, h5py, scikit-image
- cython (for C++ extensions)
- gunpowder (pipeline framework)

Additional post-processing dependencies:
- daisy (parallel processing)
- funlib.segment, funlib.evaluate
- networkx==2.2 (for MergeTree in agglomeration)
- MongoDB (optional, for large-scale RAG storage)
