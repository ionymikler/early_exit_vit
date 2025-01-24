# EEVIT: Early Exit Vision Transformer

Own implementation of [LGVIT](https://arxiv.org/abs/2308.00255). Adapted to be `torch.compile` and `torch.onnx.export` compatible.
The system allows for dynamic computational paths through the network based on confidence thresholds.

## How to Run
### Prerequisites
```bash
# Create conda environment
conda create -n eevit python=3.10
conda activate eevit
```

### Install dependencies
```bash
pip install -r requirements.txt
```
Running the Model
```bash
# Basic run: default config file, no export
python main.py 
```

### Dry run to validate configuration
```bash
python main.py --config configs/base_config.yaml --dry-run
```

### Export to ONNX format
```bash
python main.py --export-onnx
```

## Architecture Overview

The system consists of several key components:

- **Vision Transformer (EEVIT)**: Base architecture that processes image patches through self-attention layers
- **Highway Networks**: Early exit paths that can terminate computation early if confidence thresholds are met
- **Fast-Pass System**: A token-based mechanism for routing through the network

## Configuration

The system is configured through YAML files with two main sections:

### Dataset Configuration
```yaml
dataset:
  image_size: 224      # Input image dimensions
  channels_num: 3      # Number of input channels
  patch_size: 16       # Size of image patches
```

### Model Configuration
```yaml
model:
  embed_depth: 768     # Embedding dimension
  num_layers_transformer: 12  # Number of transformer layers
  num_attn_heads: 12   # Number of attention heads
  mlp_dim: 3072       # MLP hidden dimension
  pool: 'cls'         # Pooling type ('cls' or 'mean')
  dim_head: 64        # Attention head dimension
  general_dropout: 0.1 # Dropout rate
  num_classes: 1000   # Number of output classes
  
  # Early exit configuration
  early_exits:
    confidence_threshold: 0.8
    positions: [2, 5, 8]  # Layer positions for early exits
    types: ['conv1_1', 'attention', 'conv2_1']  # Types of highway networks
```

## Running the System

```bash
usage: main.py [-h] [--config-path CONFIG_PATH] [-d] [--export-onnx] [--skip-conda-env-check]

Build and run an EEVIT model, as specified in the configuration file

optional arguments:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        Path to the configuration JSON file. Default: './config/run_args.yaml'
  -d, --dry-run         Perform a dry run without making any changes
  --export-onnx, -e     Export model to ONNX format
  --skip-conda-env-check
                        Skip the check for the required conda environment
```

## Available Highway Networks

Several types of highway networks are available for early exits:

- `dummy_mlp`: Simple MLP for testing
- `conv1_1`: Single 1x1 convolution
- `conv2_1`: Double 1x1 convolution
- `attention`: Global sparse attention mechanism

## Implementation Details

The system uses PyTorch's `torch.cond` for dynamic routing through the network. Each layer includes a fast-pass token that determines whether to:
1. Skip the layer entirely (fast-pass)
2. Process through the normal computation path
3. Take an early exit through a highway network

ONNX export is supported for deployment scenarios.