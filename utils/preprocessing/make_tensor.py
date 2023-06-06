import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def padding(num):
    frames = torch.ones([num,400],dtype=torch.float16).to(device)
    return frames