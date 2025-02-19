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