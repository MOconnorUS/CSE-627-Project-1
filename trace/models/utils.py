import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from dataset import SkeletonizationDataset

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_skele_dir, val_dir, val_skele_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = SkeletonizationDataset(
        image_dir=train_dir,
        skele_dir=train_skele_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds, # THIS WILL BE LOADING THE TRAINING DATASET
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = SkeletonizationDataset(
        image_dir=val_dir,
        skele_dir=val_skele_dir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds, # THIS WILL BE LOADING THE VALIDATION DATASET
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def check_valency(preds, y_true, epoch, run_metrics):
    # Initialize dictionaries to store TP, FP, FN for each valency
    valency_metrics = {1: {'TP': 0, 'FP': 0, 'FN': 0},
                       2: {'TP': 0, 'FP': 0, 'FN': 0},
                       3: {'TP': 0, 'FP': 0, 'FN': 0},
                       4: {'TP': 0, 'FP': 0, 'FN': 0}}

    for idx in range(preds.shape[0]):  # Iterate over the batch size (32)
        pred_image = preds[idx].squeeze(0).cpu().numpy()  # Get image and remove channel dimension
        gt_image = y_true[idx].squeeze(0).cpu().numpy()  # Get ground truth image
        
        # Iterate over all pixels to calculate valency for each node
        for x in range(pred_image.shape[0]):
            for y in range(pred_image.shape[1]):
                pred_valency = get_valency(pred_image, x, y)
                gt_valency = get_valency(gt_image, x, y)
                
                # If the node exists in both the prediction and ground truth
                if pred_valency > 0 and gt_valency > 0:
                    if pred_valency == gt_valency:
                        valency_metrics[pred_valency]['TP'] += 1  # True positive
                    else:
                        valency_metrics[pred_valency]['FP'] += 1  # False positive for predicted valency
                        valency_metrics[gt_valency]['FN'] += 1  # False negative for ground truth valency
                
                # If the node is present in the prediction but not in the ground truth
                elif pred_valency > 0 and gt_valency == 0:
                    valency_metrics[pred_valency]['FP'] += 1
                
                # If the node is in the ground truth but not in the prediction
                elif pred_valency == 0 and gt_valency > 0:
                    valency_metrics[gt_valency]['FN'] += 1

    print(f'VALENCY METRICS: {valency_metrics}')
    # After calculating the metrics, calculate precision and recall for each valency
    for valency in valency_metrics:
        TP = valency_metrics[valency]['TP']
        FP = valency_metrics[valency]['FP']
        FN = valency_metrics[valency]['FN']

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        print(f"Valency {valency} Precision: {precision:.4f}, Recall: {recall:.4f}")

        if 1 == valency:
            run_metrics['Valent-1 Precision'][epoch] += precision
            run_metrics['Valent-1 Recall'][epoch] += recall
        
        if 2 == valency:
            run_metrics['Valent-2 Precision'][epoch] += precision
            run_metrics['Valent-2 Recall'][epoch] += recall
        
        if 3 == valency:
            run_metrics['Valent-3 Precision'][epoch] += precision
            run_metrics['Valent-3 Recall'][epoch] += recall

        if 4 == valency:
            run_metrics['Valent-4 Precision'][epoch] += precision
            run_metrics['Valent-4 Recall'][epoch] += recall

    run_metrics['Valent-1 Precision'][epoch] = run_metrics['Valent-1 Precision'][epoch] / 4
    run_metrics['Valent-1 Recall'][epoch] = run_metrics['Valent-1 Recall'][epoch] / 4

    run_metrics['Valent-2 Precision'][epoch] = run_metrics['Valent-2 Precision'][epoch] / 4
    run_metrics['Valent-2 Recall'][epoch] = run_metrics['Valent-2 Recall'][epoch] / 4

    run_metrics['Valent-3 Precision'][epoch] = run_metrics['Valent-3 Precision'][epoch] / 4
    run_metrics['Valent-3 Recall'][epoch] = run_metrics['Valent-3 Recall'][epoch] / 4

    run_metrics['Valent-4 Precision'][epoch] = run_metrics['Valent-4 Precision'][epoch] / 4
    run_metrics['Valent-4 Recall'][epoch] = run_metrics['Valent-4 Recall'][epoch] / 4
    
def get_valency(image, x, y):
    valency = 0
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Adjacent nodes (up, down, left, right)

    for nx, ny in neighbors:
        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:  # Check if neighbor is within bounds
            if image[nx, ny] == 1:  # If neighbor is also a node
                valency += 1
    return valency

# def check_valency(preds, y):
#     with torch.no_grad():
#         print(f'LENGTH OF PREDS: {preds.shape}')
#         print(f'LENGTH OF Y: {y.shape}')

#         print(f'TYPE OF PREDS: {type(preds)}')
#         print(f'TYPE OF Y: {type(y)}')

#         check_valency(preds)

        # for idx, item in enumerate(preds):
        #     print(f'LENGTH PER ROW IN PREDS: {item.shape}')
        #     print(f'PREDS AT ITEM: {preds[idx]}')

        #     for index, i in enumerate(item):
        #         print(f'LAYER 2: {len(i)}')
        #         print(f'LAYER 2: {item[index]}')

        #         for t, three in enumerate(i):
        #             print(f'LAYER 3: {len(three)}')
        #             print(f'LAYER 3: {i[t]}')

        # print('\n')

        # for idx, item in enumerate(y):
        #     print(f'LENGTH PER ROW IN Y: {len(item)}')
        #     print(f'Y AT ITEM: {y[idx]}')

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

def check_accuracy(loader, model, epoch, run_metrics, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0
    total_mse = 0
    model.eval()
    
    run_metrics['Accuracy'][epoch] = 0
    run_metrics['Dice Score'][epoch] = 0
    run_metrics['General Precision'][epoch] = 0
    run_metrics['General Recall'][epoch] = 0
    run_metrics['MSE'][epoch] = 0
    run_metrics['Valent-1 Precision'][epoch] = 0
    run_metrics['Valent-1 Recall'][epoch] = 0
    run_metrics['Valent-2 Precision'][epoch] = 0
    run_metrics['Valent-2 Recall'][epoch] = 0
    run_metrics['Valent-3 Precision'][epoch] = 0
    run_metrics['Valent-3 Recall'][epoch] = 0
    run_metrics['Valent-4 Precision'][epoch] = 0
    run_metrics['Valent-4 Recall'][epoch] = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            # print(f'Y IN LOADER: {y}')
            preds = torch.sigmoid(model(x))
            total_mse += mse(preds, y)

            preds = (preds > 0.5).float()
            # print(f'PREDICTIONS: {preds}')
            check_valency(preds, y, epoch, run_metrics)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum()) / ((preds + y).sum() + 1e-8)

            true_positive += ((preds == 1) & (y == 1)).sum()
            predicted_positive += (preds == 1).sum()
            actual_positive += (y == 1).sum()

    print(f'Got {num_correct}/{num_pixels} with accuracy: {num_correct/num_pixels*100:.2f}')
    print(f'Precision: {(true_positive / predicted_positive) if predicted_positive > 0 else 0}')
    print(f'Recall: {(true_positive / actual_positive) if actual_positive > 0 else 0}')
    print(f'Dice score: {dice_score/len(loader)}')
    print(f'MSE: {total_mse/len(loader)}')

    run_metrics['Accuracy'][epoch] = num_correct/num_pixels*100
    run_metrics['Dice Score'][epoch] = dice_score/len(loader)
    run_metrics['General Precision'][epoch] = (true_positive / predicted_positive) if predicted_positive > 0 else 0
    run_metrics['General Recall'][epoch] = (true_positive / actual_positive) if actual_positive > 0 else 0
    run_metrics['MSE'][epoch] = total_mse/len(loader)

    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f'{folder}/pred_{idx}.png')

            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
    
    model.train()