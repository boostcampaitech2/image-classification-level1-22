import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
from model import MyModel
from tqdm import tqdm
from PIL import ImageFile

class Trainer:
    def __init__(self, device, model, train_loader, optimizer, criterion, epoch=3):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = epoch
    
    def train(self):
        self.model.train()
        avg_loss = 0.0
        avg_acc = 0.0

        for epoch in tqdm(range(self.epoch)):
            running_loss = 0.0
            running_acc = 0.0

            pbar= tqdm(enumerate(train_loader, 0), total=len(self.train_loader), position=True, leave=True)
            for i, (images, labels) in pbar:
                inputs = images.to(self.device)
                labels = labels.to(self.device).long()
                
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_acc / len(self.train_loader.dataset)
            avg_loss += epoch_loss
            avg_acc += epoch_acc
            print(f"<<epoch-{epoch+1}>> Loss_avg : {epoch_loss:.3f}, Accuracy_avg : {epoch_acc:.3f}")

        avg_loss /= self.epoch
        avg_acc /= self.epoch
        print("Training has been finished")
        torch.save(model, f'./model_l_{avg_loss:.3f}_a_{avg_acc:.3f}.pt')

if __name__ == '__main__':
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    csv_file = '/opt/ml/input/data/train/train_with_label.csv'
    custom_dataset_train = CustomDataset(csv_file,
                                        drop_features=['id', 'race'],
                                        transform=transforms.Compose([
                                            transforms.Resize((400, 300)),
                                            transforms.ToTensor()
                                        ]),
                                        train=True)
    
    train_loader = DataLoader(custom_dataset_train,
                                batch_size=64,
                                shuffle=True,
                                num_workers=1)

    model = MyModel()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    trainer = Trainer(device, model, train_loader, optimizer, criterion, epoch=10)
    trainer.train()