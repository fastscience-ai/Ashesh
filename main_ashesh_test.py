import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from functions import *
from transformer_ashesh import *
from data_loader_one_step_UVS import load_test_data
from data_loader_one_step_UVS import load_train_data
import wandb
#torch.cuda.set_device(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand([100,3,128,128]).float().cuda()
model = Transformer(num_atoms=128, input_size=128, dim=128,
                 depth=3, heads=4, mlp_dim=512, k=64, in_channels=3)
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
tt =  torch.randint(0, 300, (1,), device=device).long()
model(x, x, tt)


