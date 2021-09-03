import os
import torch
from torch.utils.data import DataLoader
import wandb

from dataset import AlbumentationDataset
from model import MyModel
from tqdm import tqdm
import numpy as np

from GPUtil import showUtilization as gpu_usage
from datetime import datetime, timedelta, timezone
import albumentations as A

class Trainer:
    def __init__(self):
        self.is_first = True

    def make_save_dir(self):
        save_dir = '/opt/ml/code/save'
        KST = timezone(timedelta(hours=9))
        date_dir = datetime.now(KST).strftime('%y%m%d_%H:%M')
        self.save_dir = os.path.join(save_dir, date_dir)
        os.mkdir(self.save_dir)

    
    def train(self):
        EPOCHS = 4
        BATCH_SIZE = 64
        LEARNING_RATE = 1e-3

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = MyModel()
        self.model.to(self.device)

        config = {"model": self.model.name, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate" : LEARNING_RATE}
        wandb.init(project="test", entity='bagineer', config=config)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        csv_file = './train_with_label.csv'
        # self.make_save_dir()

        transform_list = [None,
                    A.HorizontalFlip(p=1),
                  A.GaussianBlur(blur_limit=(9, 9), sigma_limit=30, always_apply=True),
                  A.GaussNoise(var_limit=(100.0, 1000.0), mean=10, per_channel=True, always_apply=True),
                  A.CoarseDropout(max_holes=20, max_height=20, max_width=20,
                            min_holes=None, min_height=None, min_width=None,
                            fill_value=0, mask_fill_value=None, always_apply=True),
                  A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1,
                           drop_color=(255, 255, 255), blur_value=2, brightness_coefficient=0.7,
                           rain_type=None, always_apply=True),
                  A.MotionBlur(blur_limit=(20, 20), always_apply=True),
                  A.OpticalDistortion(distort_limit=(-0.3, 2), shift_limit=(-100, 100), interpolation=3,
                                border_mode=1, value=None, mask_value=None, always_apply=True),
                  A.RandomFog (fog_coef_lower=0.2, fog_coef_upper=0.2,
                         alpha_coef=1, always_apply=True),
                  A.RandomGridShuffle (grid=(3, 3), always_apply=True),
                  A.RandomShadow (shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2,
                            shadow_dimension=5, always_apply=True),
                  A.RandomSunFlare (flare_roi=(0, 0, 1, 1), angle_lower=0.3, angle_upper=0.5,
                              num_flare_circles_lower=6, num_flare_circles_upper=10,
                              src_radius=150, src_color=(255, 255, 255), always_apply=True),
                  A.Sharpen (alpha=(0.7, 0.7), lightness=(0.5, 1.0), always_apply=True),
                  A.Superpixels (p_replace=0.2, n_segments=100, max_size=256,
                           interpolation=1, always_apply=True),
                  A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
                  A.Cutout (num_holes=3, max_h_size=70, max_w_size=70, fill_value=0, always_apply=True),
                  A.CoarseDropout (max_holes=10, max_height=50, max_width=50, min_holes=5,
                             min_height=30, min_width=30, always_apply=True),
                  A.Downscale (scale_min=0.5, scale_max=0.5, interpolation=1, always_apply=True),
                  A.Emboss (alpha=(1, 1), strength=(1, 1), always_apply=True),
                  A.GridDistortion (num_steps=30, distort_limit=0.3, interpolation=1, border_mode=3,
                              value=None, mask_value=None, always_apply=True),
                  A.GridDropout (ratio=0.3, unit_size_min=None, unit_size_max=None, holes_number_x=3,
                           holes_number_y=3, shift_x=100, shift_y=100, random_offset=True,
                           fill_value=0, mask_fill_value=None, always_apply=True),
                  A.ImageCompression (quality_lower=10, quality_upper=50, always_apply=True)]

        trans_num = 1
        for index in range(len(transform_list) // trans_num):
            start_index = index * trans_num
            end_index = (index+1) * trans_num
            _transform = A.Compose(transform_list[start_index:end_index])
            print(_transform)

            custom_dataset_train = AlbumentationDataset(csv_file,
                                                drop_features=['id', 'race'],
                                                transform=_transform,
                                                train=True)

            train_set, valid_set = custom_dataset_train.split_balanced_dataset()
            
            train_loader = DataLoader(train_set,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        drop_last=True)

            if self.is_first:
                valid_loader = DataLoader(valid_set,
                            batch_size=256,
                            shuffle=False,
                            drop_last=True)

                self.valid_loader = valid_loader
                self.is_first = False

            self._train(train_loader)

        
    def _train(self, train_loader):
        EPOCHS = 5
        
        avg_loss = 0.0
        avg_acc = 0.0
        best_val_acc = 0

        for epoch in range(EPOCHS):
            running_loss = 0.0
            running_acc = 0.0
            self.model.train()

            # pbar= tqdm(enumerate(train_loader, 0), total=len(train_loader), position=True, leave=True)
            for images, labels in train_loader:
                inputs = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_acc += (preds == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = running_acc / len(train_loader.dataset)
            avg_loss += epoch_loss
            avg_acc += epoch_acc
            print(f"<<epoch-{epoch+1}>> Loss_avg : {epoch_loss:.3f}, Accuracy_avg : {epoch_acc:.3f}")
            wandb.log({'train_accuracy': epoch_acc, 'train_loss': epoch_loss})

            # torch.cuda.empty_cache()

            with torch.no_grad():
                self.model.eval()
                val_loss_items = []
                val_acc_items = []
                
                for images, labels in self.valid_loader:
                    inputs = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss_item = self.criterion(outputs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(self.valid_loader)
                val_acc = np.sum(val_acc_items) / len(self.valid_loader.dataset)
                print(f"<<epoch-{epoch+1}>> Valid_loss : {val_loss:.3f}, Valid_acc : {val_acc:.3f}")
                wandb.log({'validation_accuracy': val_acc, 'validation_loss': val_loss})

                if val_acc > best_val_acc:
                    # torch.save(self.model, f"{self.save_dir}/best.pt")
                    torch.save(self.model, 'best.pt')
                    best_val_acc = val_acc
                # torch.save(self.model, f"{self.save_dir}/last.pt")

                
        avg_loss /= EPOCHS
        avg_acc /= EPOCHS
        wandb.log({'train_epoch_accuracy': avg_acc, 'train_epoch_loss': avg_loss})
        print("Training has been finished")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()