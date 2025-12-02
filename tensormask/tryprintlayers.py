# Import necessary libraries
import torch
from my_tensor_mask_model import TensorMask  # Replace with the actual import statement for your model

# Define a minimal configuration (you should replace this with your actual configuration)
class MinimalConfig:
    pass

# Create an instance of the configuration and set the required attributes
cfg = MinimalConfig()
cfg.MODEL = MinimalConfig()
cfg.MODEL.TENSOR_MASK = MinimalConfig()
cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 80  # Replace with your number of classes
# Add more configuration attributes as needed

# Create an instance of the TensorMask model
tensor_mask_model = TensorMask(cfg)

# Define a function to print layer information
def print_model_layers(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Shape: {param.shape}")

# Call the function to print layer information
print_model_layers(tensor_mask_model)