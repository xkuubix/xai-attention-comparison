# %%
import os
import yaml
import uuid
import random
import numpy as np
import torch
import torch.nn as nn
import neptune
import utils
from models import GatedAttentionMIL, MultiHeadGatedAttentionMIL
from net_utils import train_gacc, validate, test, mc_test, EarlyStopping

def deactivate_batchnorm(net):
    """Deactivate BatchNorm tracking for Monte Carlo Dropout (MCDO)."""
    if isinstance(net, nn.BatchNorm2d):
        net.track_running_stats = False
        net.running_mean = None
        net.running_var = None

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    selected_device = config['device']
    device = torch.device(selected_device if torch.cuda.is_available() else "cpu")

    SEED = config['seed']
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    utils.reset_seed(SEED)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    
    run = None
    if config["neptune"]:
        run = neptune.init_run(project="ProjektMMG/MCDO")
        run["config"] = config
        run["sys/group_tags"].add(["cross-validation"])
        model_type = "SH" if config['training_plan']['criterion'].lower() == 'bce' else "MH"
        tag = f"{model_type}_{config['data']['patch_size']}_at_{config['data']['overlap']}"
        run["sys/group_tags"].add(tag)


    k_folds = config.get("k_folds", 5)
    fold_results = []


    if not os.path.exists(os.path.join(config['model_path'],tag)):
        os.makedirs(os.path.join(config['model_path'], tag))
        print(f"Creating directory for models: {os.path.join(config['model_path'], tag)}")
        
    for fold in range(k_folds):
        check_path = os.path.join(config['model_path'], tag, f"fold_{fold + 1}")
        if os.path.exists(check_path):
            print(f"Fold {fold + 1} already exists. Skipping training for this fold.")
            continue
        else:
            os.makedirs(check_path)
            print(f"Creating directory for fold {fold + 1}: {check_path}")

        print(f"\nFold {fold + 1}/{k_folds}")
        dataloaders = utils.get_fold_dataloaders(config, fold)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
        print(train_loader.dataset.df['ClassificationMapped'].value_counts())
        print(val_loader.dataset.df['ClassificationMapped'].value_counts())
        print(test_loader.dataset.df['ClassificationMapped'].value_counts())
        
        if config['training_plan']['criterion'].lower() == 'bce':
            model = GatedAttentionMIL(
                backbone=config['model'],
                feature_dropout=config['feature_dropout'],
                attention_dropout=config['attention_dropout'],
                config=config
                )
        elif config['training_plan']['criterion'].lower() == 'ce':
            model = MultiHeadGatedAttentionMIL(
                backbone=config['model'],
                feature_dropout=config['feature_dropout'],
                attention_dropout=config['attention_dropout'],
                shared_attention=config['shared_att'],
                config=config
                )
        if config["neptune"]:
            run['model/architecture'] = str(model.__class__.__name__)
        print(f"Model architecture: {str(model.__class__.__name__)}")
        
        
        model.apply(deactivate_batchnorm)
        model.to(device)

        if config['training_plan']['criterion'].lower() == 'bce':
            criterion = torch.nn.BCELoss()
        elif config['training_plan']['criterion'].lower() == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Criterion not supported")
        
        if config['training_plan']['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['training_plan']['parameters']['lr'],
                                        weight_decay=config['training_plan']['parameters']['wd'])
        elif config['training_plan']['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['training_plan']['parameters']['lr'],
                                        weight_decay=config['training_plan']['parameters']['wd'])
        else:
            raise ValueError("Optimizer not supported")
        
        early_stopping = EarlyStopping(patience=config['training_plan']['parameters']['patience'], neptune_run=run)
        utils.reset_seed(SEED)
        for epoch in range(1, config['training_plan']['parameters']['epochs'] + 1):
            train_gacc(model=model, dataloader=dataloaders['train'],
                       criterion=criterion, optimizer=optimizer, device=device, neptune_run=run, epoch=epoch,
                       accumulation_steps=config['training_plan']['parameters']['grad_acc_steps'])
            val_loss = validate(model=model, dataloader=dataloaders['val'],
                                criterion=criterion, device=device, neptune_run=run, epoch=epoch)
            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch} for fold {fold + 1}")
                break

        # model_name = os.path.join(config['model_path'], f"fold_{fold + 1}_{uuid.uuid4().hex}.pth")
        model_name = os.path.join(config['model_path'], tag, f"fold_{fold + 1}",f"{uuid.uuid4().hex}.pth")
        torch.save(early_stopping.get_best_model_state(), model_name)
        if run is not None:
            run[f"best_model_path"].log(model_name)
        if config['training_plan']['criterion'].lower() == 'bce':
            model = GatedAttentionMIL(
                backbone=config['model'],
                feature_dropout=config['feature_dropout'],
                attention_dropout=config['attention_dropout'],
                config=config
                )
        elif config['training_plan']['criterion'].lower() == 'ce':
            model = MultiHeadGatedAttentionMIL(
                backbone=config['model'],
                feature_dropout=config['feature_dropout'],
                attention_dropout=config['attention_dropout'],
                shared_attention=config['shared_att'],
                config=config
                )
        model.apply(deactivate_batchnorm)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        utils.reset_seed(SEED)
        if config['is_MCDO-test']:
            mc_test(model, test_loader, device, run, config['N'])
        else:
            test(model, test_loader, device, run)
        break # single fold per job
    if run is not None:
        run.stop()

# %%