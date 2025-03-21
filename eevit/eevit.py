# Adapted from Github's lucidrains/vit-pytorch/vit_pytorch/vit.py (3.Dec.24)
import torch
from torch import nn

from .vit_classes import PatchEmbedding, TransformerEnconder
from .utils import get_fast_pass, remove_fast_pass
from utils.arg_utils import ModelConfig
from utils.logging_utils import get_logger_ready


import torch._dynamo.config

torch._dynamo.config.capture_scalar_outputs = True  # Not sure if needed

logger = get_logger_ready("eevit.py")


class EEVIT(nn.Module):
    def __init__(self, *, config: ModelConfig, verbose: bool = False):
        # * to enforce only keyword arguments
        """
        Initializes the Vision Transformer (ViT) model.

        Args:
            image_size (int or tuple): Size of the input image. If an int is provided, it is assumed to be the size of both dimensions.
            patch_size (int or tuple): Size of the patches to be extracted from the input image. If an int is provided, it is assumed to be the size of both dimensions.
            num_classes (int): Number of output classes.
            embed_depth (int): Dimension of the embeddings.
            transformer_layers (int): Number of transformer layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the MLP (Feed-Forward) layer.
            pool (str, optional): Pooling type, either 'cls' (class token) or 'mean' (mean pooling). Default is 'cls'.
            channels (int, optional): Number of input channels. Default is 3.
            dim_head (int, optional): Depth dimension of the attention matrices in each head. Default is 64.
            dropout (float, optional): Dropout rate for the transformer. Default is 0.
            dropout_embedding (float, optional): Dropout rate for the embedding layer. Default is 0.
        """
        super().__init__()
        self.name = "EEVIT"
        logger.info("ℹ️ Initializing Vit model...")
        self.exportable = True if config.enable_export else False

        self.patch_embedding = PatchEmbedding(config, verbose=verbose)

        self.transformer = TransformerEnconder(config, verbose=verbose)

        self.pool = config.pool
        self.to_latent = nn.Identity()

        self.last_exit = nn.Linear(config.embed_depth, config.num_classes)
        self.softmax = nn.Softmax(dim=-1)

        logger.info("ViT model initialized ✅")

    def last_classifier_fw(self, x, intermediate_predictions):
        x = (
            x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        )  # take cls token or average all tokens (pooling)

        x = self.to_latent(x)  # TODO: Review why this is done in LGVIT
        logits = self.last_exit(x)

        last_layer_predictions = self.softmax(logits)

        # Adding -1.0 to indicate the model did not exit early
        last_layer_predictions = torch.cat(
            (
                last_layer_predictions,
                torch.full((1, 1), -1.0).to(last_layer_predictions.device),
            ),
            dim=1,
        )

        return last_layer_predictions

    def fast_pass(self, x, predictions: torch.Tensor):
        return predictions.clone()

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # with record_function("EEVIT.forward.embeddings"):
        embeddings = self.patch_embedding(image_tensor)

        # with record_function("EEVIT.forward.transformer"):
        x_fp, predictions = self.transformer(embeddings)

        x = remove_fast_pass(x_fp)
        fp = get_fast_pass(x_fp)
        #### CONDITIONAL ####
        if self.exportable:
            predictions = torch.cond(
                fp.any(), self.fast_pass, self.last_classifier_fw, (x, predictions)
            )
        else:
            predictions = (
                self.fast_pass(x, predictions)
                if fp.any()
                else self.last_classifier_fw(x, predictions)
            )
        #### //CONDITIONAL ####

        return predictions
