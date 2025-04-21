import torch
from sklearn import metrics
import numpy as np

def _get_trainable_params(model):
    """Get Parameters with `requires.grad` set to `True`"""
    trainable_params = []
    for x in model.parameters():
        if x.requires_grad:
            trainable_params.append(x)
    return trainable_params


def _train_model(model, train_loader, epoch, num_epochs, optimizer, criterion, writer, current_lr, log_every=100, use_regression=True):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = [img.cuda() for img in images]
            labels = labels.cuda()

        outputs = model(images)  # shape: [batch_size, 1]

        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy by rounding predictions
        preds = outputs.detach().round().clamp(0, 3).squeeze(1).long()  # [batch_size]
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        if (i % log_every == 0) and (i > 0):
            print(f"[Epoch {epoch+1}/{num_epochs} | Batch {i}/{len(train_loader)}] "
                  f"Avg Train Loss: {total_loss / (i + 1):.4f} | "
                  f"Train Acc: {total_correct / total_samples:.4f} | "
                  f"LR: {current_lr}")

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = total_correct / total_samples

    writer.add_scalar('Train/Loss_epoch', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy_epoch', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def _evaluate_model(model, val_loader, criterion, epoch, num_epochs, writer, current_lr, log_every=20, use_regression=True):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = [img.cuda() for img in images]
                labels = labels.cuda()

            outputs = model(images)  # shape: [batch_size, 1]

            loss = criterion(outputs, labels.float().unsqueeze(1))
            total_loss += loss.item()

            # Compute accuracy
            preds = outputs.round().clamp(0, 3).squeeze(1).long()
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            if (i % log_every == 0) and (i > 0):
                print(f"[Epoch {epoch+1}/{num_epochs} | Batch {i}/{len(val_loader)}] "
                      f"Avg Val Loss: {total_loss / (i + 1):.4f} | "
                      f"Val Acc: {total_correct / total_samples:.4f} | "
                      f"LR: {current_lr}")

    epoch_loss = total_loss / len(val_loader)
    epoch_acc = total_correct / total_samples

    writer.add_scalar('Val/Loss_epoch', epoch_loss, epoch)
    writer.add_scalar('Val/Accuracy_epoch', epoch_acc, epoch)

    print(f"Epoch {epoch+1} End | Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']