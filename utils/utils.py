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

def _evaluate_model(model, val_loader, criterion, epoch, num_epochs, writer, current_lr, log_every=20):
    """Runs model over val dataset and returns accuracy and avg val loss"""

    model.eval()
    y_preds = []
    y_gt = []
    losses = []

    with torch.no_grad():
        for i, (images, label) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = [image.cuda() for image in images]
                label = label.cuda().long()
                if label.dim() == 0:
                    label = label.unsqueeze(0)


            output = model(images)
            loss = criterion(output, label)
            loss_value = loss.item()
            losses.append(loss_value)

            # Convert output to predicted class
            pred_classes = torch.argmax(output, dim=1)

            # Store predictions and true labels
            y_gt.extend(label.cpu().numpy().tolist())
            y_preds.extend(pred_classes.cpu().numpy().tolist())

            try:
                acc = accuracy_score(y_gt, y_preds)
            except:
                acc = 0.0

            writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
            writer.add_scalar('Val/Accuracy', acc, epoch * len(val_loader) + i)

            if (i % log_every == 0) and (i > 0):
                print(f'''[Epoch: {epoch + 1} / {num_epochs} | Batch : {i} / {len(val_loader)} ]| Avg Val Loss: {np.mean(losses):.4f} | Val Accuracy: {acc:.4f} | lr: {current_lr}''')

    writer.add_scalar('Val/Accuracy_epoch', acc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_acc_epoch = np.round(acc, 4)

    return val_loss_epoch, val_acc_epoch

def _train_model(model, train_loader, epoch, num_epochs, optimizer, criterion, writer, current_lr, log_every=100):
    model.train()

    y_preds = []
    y_gt = []
    losses = []

    for i, (images, label) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda().long()
            if label.dim() == 0:
                label = label.unsqueeze(0)

        output = model(images)  # shape: [batch_size, 4]

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Convert logits to class predictions
        pred_classes = torch.argmax(output, dim=1)

        y_gt.extend(label.cpu().numpy().tolist())
        y_preds.extend(pred_classes.cpu().numpy().tolist())

        try:
            acc = accuracy_score(y_gt, y_preds)
        except:
            acc = 0.0

        writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Train/Accuracy', acc, epoch * len(train_loader) + i)

        if (i % log_every == 0) and (i > 0):
            print(f'''[Epoch: {epoch + 1} / {num_epochs} | Batch : {i} / {len(train_loader)} ]| Avg Train Loss: {np.mean(losses):.4f} | Accuracy: {acc:.4f} | lr: {current_lr}''')

    writer.add_scalar('Train/Accuracy_epoch', acc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_acc_epoch = np.round(acc, 4)

    return train_loss_epoch, train_acc_epoch

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']