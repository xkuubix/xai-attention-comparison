import torch
import copy
from sklearn.metrics import classification_report
import torch.nn as nn


def deactivate_batchnorm(model):
    if isinstance(model, nn.BatchNorm2d):
        model.track_running_stats = False
        model.running_mean = None
        model.running_var = None

def train_gacc(model, dataloader, criterion, optimizer, device, neptune_run=None, epoch=100, accumulation_steps=8, fold_idx=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        images, targets = batch['image'].to(device), batch['target']['label'].to(device)
        
        if str(model.__class__.__name__) == "CLAM":
            output, _ = model(images, targets, instance_eval=True)
        else:
            output, _ = model(images)
        
        if str(model.__class__.__name__) == "GatedAttentionMIL":
            output = torch.sigmoid(output[0])
        
        if str(model.__class__.__name__) == "DSMIL":
            classes, bag_prediction = output
            max_prediction, index = torch.max(classes, 0)
            output = torch.sigmoid(bag_prediction)
            loss_bag = criterion(output, targets.view(1, -1))
            loss_max = criterion(torch.sigmoid(max_prediction.view(1, -1)), targets.view(1, -1))
            loss_total = 0.5*loss_bag + 0.5*loss_max
            loss = loss_total.mean()
        elif "GatedAttentionMIL" in str(model.__class__.__name__):
            loss = criterion(output, targets)
        elif str(model.__class__.__name__) == "CLAM":
            loss = criterion(output[1], targets)
            c1 = 0.7
            c2 = 1 - c1
            loss = loss * c1 + output[-1]['instance_loss'] * c2 
            output = output[1]

        running_loss += loss.item()
        
        loss = loss / accumulation_steps
        
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        if str(model.__class__.__name__) in ["GatedAttentionMIL", "DSMIL"]:
            preds = (output.view(-1) > 0.5).float()
        elif str(model.__class__.__name__) in ["MultiHeadGatedAttentionMIL", "CLAM"]:
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



def validate(model, dataloader, criterion, device, neptune_run=None, epoch=100, fold_idx=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch['image'].to(device), batch['target']['label'].to(device)
            
            if str(model.__class__.__name__) == "CLAM":
                output, _ = model(images, targets, instance_eval=True)
            else:
                output, _ = model(images)
            if str(model.__class__.__name__) == "GatedAttentionMIL":
                output = torch.sigmoid(output[0])
    
            if str(model.__class__.__name__) == "DSMIL":
                classes, bag_prediction = output
                max_prediction, index = torch.max(classes, 0)
                output = torch.sigmoid(bag_prediction)
                loss_bag = criterion(output, targets.view(1, -1))
                loss_max = criterion(torch.sigmoid(max_prediction.view(1, -1)), targets.view(1, -1))
                loss_total = 0.5*loss_bag + 0.5*loss_max
                loss = loss_total.mean()
            elif "GatedAttentionMIL" in str(model.__class__.__name__):
                loss = criterion(output, targets)
            elif str(model.__class__.__name__) == "CLAM":
                loss = criterion(output[1], targets)
                c1 = 0.7
                c2 = 1 - c1
                loss = loss * c1 + output[-1]['instance_loss'] * c2
                output = output[1]

            
            running_loss += loss.item()

            if str(model.__class__.__name__) in ["GatedAttentionMIL", "DSMIL"]:
                preds = (output.view(-1) > 0.5).float()
            elif str(model.__class__.__name__) in ["MultiHeadGatedAttentionMIL", "CLAM"]:
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


def test(model, dataloader, device, neptune_run=None, fold_idx=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch['image'].to(device), batch['target']['label'].to(device)
            if str(model.__class__.__name__) == "CLAM":
                output, _ = model(images, targets)
            else:
                output, _ = model(images)
            
            if str(model.__class__.__name__) == "DSMIL":
                output = torch.sigmoid(output[1])
            elif str(model.__class__.__name__) == "CLAM":
                output = output[1]
            elif str(model.__class__.__name__) == "GatedAttentionMIL":
                output = torch.sigmoid(output[0])
            
            if str(model.__class__.__name__) in ["GatedAttentionMIL", "DSMIL"]:
                preds = (output.view(-1) > 0.5).float()
            elif str(model.__class__.__name__) in ["MultiHeadGatedAttentionMIL", "CLAM"]:
                preds = output.argmax(dim=1)
            
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


def mc_test(model, dataloader, device, neptune_run=None, fold_idx=None, N=50):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch['image'], batch['target']['label'].to(device)
            output, _ = model.mc_inference(x=images, N=N)
            # output, _ = model(images, N=N)
            if str(model.__class__.__name__) == "GatedAttentionMIL":
                output = torch.sigmoid(output.squeeze(0))
                preds = (output.view(-1) > 0.5).float()
            elif str(model.__class__.__name__) == "MultiHeadGatedAttentionMIL": 
                output = torch.nn.functional.softmax(output, dim=-1)
                mc_output_mean = output.mean(dim=0)
                preds = mc_output_mean.argmax(dim=1)
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
