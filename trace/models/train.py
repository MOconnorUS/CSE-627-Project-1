import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs

# Hyper Params
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '../../data/train_images/'
TRAIN_SKELE_DIR = '../../data/train_skele_images/'
VAL_IMG_DIR = '../../data/val_images/'
VAL_SKELE_DIR = '../../data/val_skele_images/'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_SKELE_DIR,
        VAL_IMG_DIR,
        VAL_SKELE_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    run_metrics = {
        'Accuracy': {},
        'General Precision': {},
        'General Recall': {},
        'Dice Score': {},
        'MSE': {},
        'Valent-1 Precision': {},
        'Valent-1 Recall': {},
        'Valent-2 Precision': {},
        'Valent-2 Recall': {},
        'Valent-3 Precision': {},
        'Valent-3 Recall': {},
        'Valent-4 Precision': {},
        'Valent-4 Recall': {}
    }


    scaler = torch.amp.GradScaler(device=DEVICE)
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        save_checkpoint(checkpoint)
        
        # check accuracy
        check_accuracy(val_loader, model, epoch, run_metrics, device=DEVICE)

        # print some examples
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

        # check_loader(val_loader, DEVICE)

    print(f'Average Accuracy: {sum(run_metrics["Accuracy"].values()) / len(run_metrics['Accuracy'])}')
    print(f'Peak Accuracy: {max(run_metrics["Accuracy"].values())}')

    print(f'Average General Precision: {sum(run_metrics["General Precision"].values()) / len(run_metrics['General Precision'])}')
    print(f'Peak General Precision: {max(run_metrics["General Precision"].values())}')

    print(f'Average General Recall: {sum(run_metrics["General Recall"].values()) / len(run_metrics['General Recall'])}')
    print(f'Peak General Recall: {max(run_metrics["General Recall"].values())}')

    print(f'Average Dice Score: {sum(run_metrics["Dice Score"].values()) / len(run_metrics['Dice Score'])}')
    print(f'Peak Dice Score: {max(run_metrics["Dice Score"].values())}')

    print(f'Average MSE: {sum(run_metrics["MSE"].values()) / len(run_metrics['MSE'])}')
    print(f'Peak MSE: {max(run_metrics["MSE"].values())}')

    print(f'Average Valent-1 Precision: {sum(run_metrics["Valent-1 Precision"].values()) / len(run_metrics['Valent-1 Precision'])}')
    print(f'Peak Valent-1 Precision: {max(run_metrics["Valent-1 Precision"].values())}')

    print(f'Average Valent-1 Recall: {sum(run_metrics["Valent-1 Recall"].values()) / len(run_metrics['Valent-1 Recall'])}')
    print(f'Peak Valent-1 Recall: {max(run_metrics["Valent-1 Recall"].values())}')

    print(f'Average Valent-2 Precision: {sum(run_metrics["Valent-2 Precision"].values()) / len(run_metrics['Valent-2 Precision'])}')
    print(f'Peak Valent-2 Precision: {max(run_metrics["Valent-2 Precision"].values())}')

    print(f'Average Valent-2 Recall: {sum(run_metrics["Valent-2 Recall"].values()) / len(run_metrics['Valent-2 Recall'])}')
    print(f'Peak Valent-2 Recall: {max(run_metrics["Valent-2 Recall"].values())}')

    print(f'Average Valent-3 Precision: {sum(run_metrics["Valent-3 Precision"].values()) / len(run_metrics['Valent-3 Precision'])}')
    print(f'Peak Valent-3 Precision: {max(run_metrics["Valent-3 Precision"].values())}')

    print(f'Average Valent-3 Recall: {sum(run_metrics["Valent-3 Recall"].values()) / len(run_metrics['Valent-3 Recall'])}')
    print(f'Peak Valent-3 Recall: {max(run_metrics["Valent-3 Recall"].values())}')

    print(f'Average Valent-4 Precision: {sum(run_metrics["Valent-4 Precision"].values()) / len(run_metrics['Valent-4 Precision'])}')
    print(f'Peak Valent-4 Precision: {max(run_metrics["Valent-4 Precision"].values())}')

    print(f'Average Valent-4 Recall: {sum(run_metrics["Valent-4 Recall"].values()) / len(run_metrics['Valent-4 Recall'])}')
    print(f'Peak Valent-4 Recall: {max(run_metrics["Valent-4 Recall"].values())}')

if __name__ == '__main__':
    main()