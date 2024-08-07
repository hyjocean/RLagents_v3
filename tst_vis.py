import torch

model = torch.load(f='params/model_params_b1.pth')
st_dict = model.state_dict()
print(model.state_dict())