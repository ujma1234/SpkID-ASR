import importlib
import torch
import os
from utils.config_utils import load_config, convert_to_int
from torch import nn
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available else "cpu"

class BaseModel:
    def __init__(self, config_path, module_name, model_name):
        self.config = load_config(config_path)
        self.module_name = module_name
        self.model_name = model_name

        model_module = importlib.import_module(f"models.{module_name}")
        model_class = getattr(model_module, model_name)

        self.config = convert_to_int(self.config[module_name][model_name])

        self.model = model_class(*self.config).to(device)
    
    def load_ckpt(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(ckpt["model_state_dict"])
    
    def fit(self, train_loader, lr=0.0001, epoch_num=10, save_path=None):
        if save_path is None:
            raise "save path is None"
        try:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        except OSError:
            print("Error: Failed make directory")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        tqdm_size = 60
        max_loss = 50000

        def calc_accuracy(X,Y):
            _, index = torch.max(X, 1)
            train_acc = (index == Y).sum().data.cpu().numpy()/index.size()[0]
            return train_acc

        for epoch in range(epoch_num):
            self.model.train()
            train_acc = 0.0
            total_loss = 0
            for _, (x, labels) in enumerate(tqdm(train_loader, total = len(train_loader), ncols=tqdm_size)):
                optimizer.zero_grad()
                
                x = x.to(device)
                logits = self.model(x).to(device)
                labels = labels.to(device)
                loss = criterion(logits, labels)
                train_acc += calc_accuracy(logits, labels)

                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.6)

                optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / len(train_loader)
            
            if epoch_loss < max_loss:
                print(epoch_loss, train_acc/len(train_loader))
                torch.save(
                    {
                        "model":"Simple_classification",
                        "epoch":epoch,
                        "model_state_dict":self.model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "description":f"Simple_classification",
                    },
                    f"{save_path}/{self.module_name}-{self.model_name}.pt",
                )
                max_loss = epoch_loss
                
            
                
    def predict(self, input):
        input = input.to(device)
        softmax = torch.nn.Softmax(dim=-1)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input)
            logits = softmax(logits)
            p, index = torch.max(logits, 1)
            for num in range(p.size()[0]):
                print(f'가장 높은 확률: {int(p[num] * 100)}%, 예측 정답: {int(index[num])}')
        return p, index
        
