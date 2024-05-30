# 2-dim integer relu

import torch

# fix random seed
torch.manual_seed(42)

# Define the model
class ReLU(torch.nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
    def forward(self, x):
        return torch.nn.functional.relu(x)
    
# Create the model
model = ReLU()
model.eval()

input = torch.tensor([[-1.0,1.0]], dtype=torch.float32)
torch.onnx.export(model, input, "model.onnx", export_params=True, opset_version=7, do_constant_folding=True, input_names = ['input'], output_names = ['output'], verbose=False)

# Export sample input and output
output = model(input)

# save input.bin
input = input.detach().numpy()
input.tofile("input.bin")
print("Input: 0x"+input.data.hex())

# save output.bin
output = output.detach().numpy()
output.tofile("output.bin")
print("Output: 0x"+output.data.hex())