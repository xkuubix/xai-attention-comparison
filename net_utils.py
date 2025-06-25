import torch
import copy
from sklearn.metrics import classification_report
import torch.nn as nn


def deactivate_batchnorm(model):
    if isinstance(model, nn.BatchNorm2d):
        model.track_running_stats = False
        model.running_mean = None
        model.running_var = None

def train(model, dataloader, criterion, optimizer, device, neptune_run, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        images, targets = batch['image'].to(device), batch['target']['label'].to(device)
        for param in model.parameters():
            param.grad = None
        outputs, _ = model(images)
        output = torch.sigmoid(outputs.squeeze(0))
        loss = criterion(output, targets.unsqueeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = (output.view(-1) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    if neptune_run is not None:
        neptune_run["train/epoch_loss"].log(epoch_loss)
        neptune_run["train/epoch_acc"].log(epoch_acc)
    print(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


def train_gacc(model, dataloader, criterion, optimizer, device, neptune_run, epoch, accumulation_steps=8, fold_idx=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        images, targets = batch['image'], batch['target']['label'].to(device)
        output, _ = model(images)
        # output = torch.sigmoid(outputs.squeeze(0))
        loss = criterion(output, targets)
       
        running_loss += loss.item()
        
        loss = loss / accumulation_steps
        
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        # preds = (output.view(-1) > 0.5).float()
        preds = output.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    if not fold_idx:
        if neptune_run is not None:
            neptune_run["train/epoch_loss"].log(epoch_loss)
            neptune_run["train/epoch_acc"].log(epoch_acc)
    else:
        if neptune_run is not None:
            neptune_run[f"{fold_idx}/train/epoch_loss"].log(epoch_loss)
            neptune_run[f"{fold_idx}/train/epoch_acc"].log(epoch_acc)
    
    print(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")



def validate(model, dataloader, criterion, device, neptune_run, epoch, fold_idx=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch['image'].to(device), batch['target']['label'].to(device)
            output, _ = model(images)
            # output = torch.sigmoid(outputs.squeeze(0))
            loss = criterion(output, targets)
            running_loss += loss.item()
            # preds = (output.view(-1) > 0.5).float()
            preds = output.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    if not fold_idx:
        if neptune_run is not None:
            neptune_run["val/epoch_loss"].log(epoch_loss)
            neptune_run["val/epoch_acc"].log(epoch_acc)
    else:
        if neptune_run is not None:
            neptune_run[f"{fold_idx}/val/epoch_loss"].log(epoch_loss)
            neptune_run[f"{fold_idx}/val/epoch_acc"].log(epoch_acc)
        
    print(f"Epoch {epoch} - Val Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return epoch_loss


def test(model, dataloader, device, neptune_run, fold_idx=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch['image'].to(device), batch['target']['label'].to(device)
            output, _, _ = model(images)
            # output = torch.sigmoid(outputs.squeeze(0))
            preds = output.argmax(dim=1)
            # preds = (output.view(-1) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_acc = correct / total
    report = classification_report(all_targets, all_preds, target_names=["Negative", "Positive"])
    if not fold_idx:
        if neptune_run is not None:
            neptune_run["test/accuracy"] = test_acc
            neptune_run["test/classification_report"] = report
    else:
        if neptune_run is not None:
            neptune_run[f"test/accuracy_fold{fold_idx}"] = test_acc
            neptune_run[f"test/classification_report_fold{fold_idx}"] = report

    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:\n", report)
    return test_acc, report


def mc_test(model, dataloader, device, neptune_run, fold_idx=None, N=50):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch['image'].to(device), batch['target']['label'].to(device)
            output, _, _ = model.mc_inference(input_tensor=images, N=N, device=device)
            # output, _ = model(images, N=N)
            output = torch.nn.functional.softmax(output, dim=-1)
            mc_output_mean = output.mean(dim=0)
            # output = torch.sigmoid(outputs.squeeze(0))
            preds = mc_output_mean.argmax(dim=1)
            # preds = (output.view(-1) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_acc = correct / total
    report = classification_report(all_targets, all_preds, target_names=["Negative", "Positive"])
    if not fold_idx:
        if neptune_run is not None:
            neptune_run["test/accuracy"] = test_acc
            neptune_run["test/classification_report"] = report
    else:
        if neptune_run is not None:
            neptune_run[f"test/accuracy_fold{fold_idx}"] = test_acc
            neptune_run[f"test/classification_report_fold{fold_idx}"] = report

    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:\n", report)
    return test_acc, report

class EarlyStopping:
    def __init__(self, patience=5, neptune_run=None):
        self.patience = patience
        self.counter = patience
        self.best_loss = float('inf')
        self.best_model_state = None
        self.neptune_run = neptune_run

    def __call__(self, current_loss, model):
        copy_model = False
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = self.patience
            copy_model = True
        else:
            self.counter -= 1

        if self.neptune_run is not None:
            self.neptune_run["val/patience_counter"].log(self.counter)

        if copy_model:
            self.best_model_state = copy.deepcopy(model.state_dict())

        return not self.counter

    def get_best_model_state(self):
        """Return the best model state dictionary."""
        return self.best_model_state
