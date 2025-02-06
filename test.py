from utils import get_config_dict, get_model
from utils.arg_utils import parse_config_dict
from utils.model_utils import load_pretrained_weights

config = get_config_dict()
model_config = parse_config_dict(config["model"].copy())
model = get_model(model_config, verbose=True)

model, incompatible_keys = load_pretrained_weights(
    model=model,
    pretrained_path="/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100/pytorch_model.bin",
    config_pretrained_path="/home/iony/DTU/f24/thesis/code/lgvit/LGViT-ViT-Cifar100/config.json",
    verbose=True,  # Set to True to see incompatible keys
)
