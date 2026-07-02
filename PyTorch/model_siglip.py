# PaliGemma (often misspelled as "polygama") is a family of lightweight, open vision-language models (VLMs) developed by Google. 
# It comes in different sizes, including PaliGemma-7B, PaliGemma-13B, and PaliGemma-34B.
# It come in 224, 448, and 896 image sizes. The model is designed to be efficient and effective for various vision-language tasks, such as image captioning, visual question answering, and image-text retrieval.


# import stuffs which are needed for the model
from typing import Optional, Tuple
import torch
import torch.nn as nn

#Paligemma comes in different sizes, we can define a config class to specify the size of the model we want to use.
#Config class for the Siglip model, which contains hyperparameters and settings for the model architecture
class SiglipVisionConfig:
    
    def __init__(
            self,
            hiden_size: int = 768, #size of the embedding vector of this vision encoder model
            intermediate_size: int = 3072, #size of the intermediate layer in the feedforward network
            num_hidden_layers: int = 12, #number of layers in the vision transformer encoder
            num_attention_heads: int = 12, #number of attention heads in the multi-head attention mechanism
            num_channels: int = 3, #number of input channels (e.g., 3 for RGB images)
            image_size: int = 224, #size of the input image (height and width)
            patch_size: int = 16, #size of the patches extracted from the input image (16x16 pixels)
            layer_norm_eps: float = 1e-6, #epsilon value for layer normalization
            attention_dropout: float = 0.0, #dropout rate for the attention mechanism (we will not use dropout in the attention mechanism, so 0.0 is a reasonable choice)
            dropout_rate: float = 0.1, #dropout rate for regularization
            num_image_tokens: int = None, #number of contextual embeddings for the image (calculated based on image size and patch size)
            **kwargs
            ):
        super().__init__()
        self.hiden_size = hiden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.dropout_rate = dropout_rate
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    # This class is responsible for extracting patches from the input image, flattening them, and converting them into embeddings.
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size #size of the embedding vector for each image
        self.image_size = config.image_size #how big is the input image (height and width)
        self.patch_size = config.patch_size #how big i sthe patch size (height and width) that we will extract from the input image

        # Patch extraction with 2d convolution, which will extract features from the input image patch by patch (there is no overlap) and convert them into embeddings.
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels, #how many channels are in the input image, which is equal to the number of channels in the input image (e.g., 3 for RGB images)
            out_channels=self.embed_dim, #how many kernals depends on how many output channels we want, which is equal to the embedding size of the model
            kernel_size=self.patch_size, 
            stride=self.patch_size, #Stride is equal to the patch size, which means that we will extract non-overlapping patches from the input image. 
            padding="valid", # This indicates "no padding is added to the input image" before applying the convolution operation. The convolution will be applied only to the valid regions of the input image, and any pixels that do not fit into a complete patch will be ignored.
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches #how many positional encoding we need -> equals to the number of patches we have extracted from the input image. Each patch will have a unique positional encoding.
        
        # Positional encoding is a technique used to inject information about the position of each patch in the input image into the embeddings. This is important because the transformer architecture does not have any inherent notion of order or position, so we need to provide this information explicitly.
        # The positional encoding is implemented as a learnable embedding layer, which will learn a unique embedding vector for each position in the input image. The dimension of the positional encoder is the same as the patch vector size, which is equal to the embedding size of the model.
        # Each positional embedding vector will be added to the corresponding patch embedding vector, which will allow the model to learn the relationships between patches based on their positions in the input image.
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim) # dimension of positional encoder is same as the patch vector size, which is equal to the embedding size of the model.
        
        # Registering the position_ids buffer, which is a tensor that contains the position indices for each patch in the input image. 
        # This buffer is not a learnable parameter, but it is used to index into the positional embedding layer to retrieve the corresponding positional embeddings for each patch.
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
        #self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]. Height and width of the image will be 224*224 as the PaliGemma model is trained on 224*224 images.
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W, each patch is a vetor of size [Embed_Dim]

        # We flatten the patches to get a tensor where first element of the tensor is first patch and last element is the last patch.
        embeddings = patch_embeds.flatten(2)  
        
        # Transpose to get NumPatches to come before Embed_Dim, so that it becomes a batch of sequence of embeddings.
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
    
class SiglipEncoderLayer(nn.Module):
    # This class is responsible for running the batch of list of embeddings through the transformer encoder layers, which will output a batch of list of contextual embeddings.
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class SigLipVisionTransformer(nn.Module):
    # It is a torch layer where we pass the configuration of the model and it will create the vision transformer model based on the configuration.
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hiden_size

        # We first need to extract the patches from the image and converts them into embeddings with the below layer/class.
        self.embeddings = SiglipVisionEmbeddings(config)

        # After patch extraction we will run it through a list of layers, which will be the transformer encoder layers.
        # We will create a list of layers based on the number of hidden layers specified in the config.
        self.encoder = SiglipEncoder(config)

        # After the encoder we will apply a layer normalization to the output of the encoder.
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, num_image_tokens, hidden_size] -> (Batch_size, Num_Patches, Embed_Dim)

        # Extract patches (using convolution) from the input image, flatten them and convert them into embeddings using the embedding layer, and add the positional encoding to the embeddings.
        # The output of this layer will be a batch of list of embeddings. One list of embeddings for each image.
        hidden_states = self.embeddings(pixel_values)

        # We take this batch of list of embeddings and run it through the "transformer encoder" layers, which will output a batch of list of contextual embeddings.
        # One list of contextual embeddings for each image.
        last_hidden_state = self.encoder(hidden_states)

        # After the encoder we will apply a layer normalization to the output of the encoder.
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):
    # This vision model will take batch of images as input and will output the contextual embeddings for each image in the batch. 
    # The output will be a batch of list of embeddings. One list of embeddings for each image.
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)

    # Pixel values are the input images, which are expected to be in the shape of (batch_size, num_channels, height, width)
    def forward(self, pixel_values) -> Tuple:
        # pixel_values: [batch_size, num_image_tokens, hidden_size] -> (Batch_size, Num_Patches, Embed_Dim)
        # Pixel values of the input images are loaded with numpy, when loaded with numpy it gets converted into an array i.e. channel*height*width, but we have a batch of images that's why we have a batch size here. 
        # Our vision transformer converts pixel values into batch sizes, number of patches, and embedding size.

        return self.vision_model(pixel_values=pixel_values)

class ModelIndSigLip(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ModelIndSigLip, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x    
    
    class ModelIndSigLipWithLipschitz(ModelIndSigLip):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lipschitz_constant: Optional[float] = None):
            super().__init__(input_dim, hidden_dim, output_dim)
            self.lipschitz_constant = lipschitz_constant

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = super().forward(x)
            if self.lipschitz_constant is not None:
                x = torch.clamp(x, min=-self.lipschitz_constant, max=self.lipschitz_constant)
            return x
        
        class ModelIndSigLipWithLipschitzAndDropout(ModelIndSigLipWithLipschitz):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lipschitz_constant: Optional[float] = None, dropout_rate: float = 0.5):
                super().__init__(input_dim, hidden_dim, output_dim, lipschitz_constant)
                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = super().forward(x)
                x = self.dropout(x)
                return x    
            
class SigLipModelConfig:
    def __init__(self, input_dim: int, hidden_dim: int = 768, output_dim: int, lipschitz_constant: Optional[float] = None, dropout_rate: float = 0.5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lipschitz_constant = lipschitz_constant
        self.dropout_rate = dropout_rate

    def create_model(self) -> ModelIndSigLipWithLipschitzAndDropout:
        return ModelIndSigLipWithLipschitzAndDropout(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            lipschitz_constant=self.lipschitz_constant,
            dropout_rate=self.dropout_rate
        )