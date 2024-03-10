import torch
import torch.nn as nn
import random
import math
import config
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm

def multi_scale(image, target_shape, max_stride):
    # compute different sizes of the image (divisible by 32) to train, just like MTCNN
    size = random.randrange(target_shape * 0.5, target_shape + max_stride) // max_stride * max_stride
    sf = size / max(image.shape[2:])
    h, w = image.shape[2:]
    ns = [math.ceil(i * sf / max_stride) * max_stride for i in [h, w]]
    image = nn.functional.interpolate(image, size=ns, mode='bilinear', align_corners=False)

    return image

def get_loaders(root_dir, batch_size, num_classes=len(config.COCO), rect_training=False, box_format='coco'):
    S = [8, 16, 32]
    train_augmentations = config.TRAIN_TRANSFORMS
    val_augmenttations = None

    train_dataset = Training_Dataset(root_dir, train_augmentations, True, rect_training)
    val_dataset = Validation_Dataset(root_dir, config.ANCHORS, val_augmenttations, False, S, rect_training)

    train_loader = DataLoader(train_dataset, batch_size, True)
    val_loader = DataLoader(val_dataset, batch_size, False)

    return train_loader, val_loader

def train(model, loader, optimizer, loss_fn, scaler, epoch, num_epochs, multi_scale_training=True):
    print(f"Training epoch {epoch}/{num_epochs}")
    batch_size = 32
    accumulate = 1
    last_opt_step = -1

    loop = tqdm(loader)
    loss_epoch = 0
    nb = len(loader)
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loop):
        images /= 255.0
        if multi_scale_training:
            images = multi_scale(images, 640, 32)

        images = images.to(config.DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)
            loss = loss_fn(out, targets)
            loss_epoch += loss

        scaler.scale(loss).backward()

        if batch_idx - last_opt_step >= accumulate or (batch_idx == nb-1):
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            last_opt_step = batch_idx

        freq = 10
        if batch_idx % freq == 0:
            loop.set_postfix(average_loss_batches=avg_batches_loss.item() / freq)
            avg_batches_loss = 0

        print(f"==> training_loss: {(loss_epoch.item() / nb):.2f}")