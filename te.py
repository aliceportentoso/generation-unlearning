import torch
from transformers import ClapModel

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
torch.save(model.state_dict(), "clap-htsat-unfused.pt")