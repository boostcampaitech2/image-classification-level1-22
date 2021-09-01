from torch.utils.data import Subset,Dataset
from torch.utils.data import DataLoader
import dataset_

def getDataloader(train_idx, val_idx, batch_size, num_workers):
    train_set = dataset_.CDataset(train=True)
    val_set = dataset_.CDataset(train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=num_workers,drop_last=True,
    shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
    drop_last=True, shuffle=False)
    #print(len(train_loader))
    return train_loader, val_loader
