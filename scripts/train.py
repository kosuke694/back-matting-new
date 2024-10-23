import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YourModel
from dataset import InstrumentDataset
from utils import save_checkpoint, load_checkpoint, Logger

def train(model, dataloader, optimizer, criterion, epoch, device, log_interval=10):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}')
    return running_loss / len(dataloader)

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['model_save_dir'], exist_ok=True)
    logger = Logger(config['log_dir'])

    train_dataset = InstrumentDataset(config['train_data_dir'], config['train_mask_dir'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = YourModel().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    start_epoch = 0
    if config.get('resume_checkpoint'):
        start_epoch = load_checkpoint(config['resume_checkpoint'], model, optimizer)

    for epoch in range(start_epoch, config['epochs']):
        loss = train(model, train_loader, optimizer, criterion, epoch, device)
        logger.log(epoch, loss)
        if epoch % config['checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, config['model_save_dir'])

    print("Training complete!")

if __name__ == "__main__":
    config_path = "configs/train_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    main(config)
