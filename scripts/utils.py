import os
import torch

def save_checkpoint(model, optimizer, epoch, save_dir):
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] + 1

class Logger:
    def __init__(self, log_dir):
        self.log_file = os.path.join(log_dir, "train.log")
    
    def log(self, epoch, loss):
        with open(self.log_file, "a") as f:
            f.write(f"Epoch: {epoch}, Loss: {loss}\n")
        print(f"Epoch: {epoch}, Loss: {loss}")
