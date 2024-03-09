import torch
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
EPOCHS = 100
NUM_WORKERS = 8
CHECKPOINT_FILE = 'b3.pth.tar'
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomCrop(size=(120, 120)),
    transforms.Normalize(
        mean=[0.3199, 0.224, 0.1609],
        std=[0.302, 0.2183, 0.1741],
    ),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.Normalize(
        mean=[0.3199, 0.224, 0.1609],
        std=[0.302, 0.2183, 0.1741],
    ),
    transforms.ToTensor()
])