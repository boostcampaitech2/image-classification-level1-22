from torchvision import transforms

def get_transfrom():
    transform = transforms.Compose([
        #transforms.CenterCrop(350),
        #transforms.Resize((224, 224)), # for ViT
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])
    return transform
