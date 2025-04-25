import torch
import torch.onnx
from test_fruit_model import TestModel

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TestModel().to(device)
model.load_state_dict(torch.load('Fruits_edible.pt', map_location=device))
model.eval()

# Create a sample input tensor
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# Export the model to ONNX format
torch.onnx.export(
    model,                       # model being run
    dummy_input,                 # model input (or a tuple for multiple inputs)
    "Fruits_edible.onnx",        # where to save the model
    export_params=True,          # store the trained parameter weights inside the model file
    opset_version=12,            # the ONNX version to export the model to
    do_constant_folding=True,    # whether to execute constant folding for optimization
    input_names=['input'],       # the model's input names
    output_names=['fruit_type', 'freshness'],  # the model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},
        'fruit_type': {0: 'batch_size'},
        'freshness': {0: 'batch_size'}
    }
)

print("Model has been converted to ONNX format and saved as 'Fruits_edible.onnx'")
