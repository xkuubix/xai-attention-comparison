# %%
import pickle, yaml, logging
import os, random, uuid, numpy as np
import torch
import neptune
from models import GatedAttentionMIL, MultiHeadGatedAttentionMIL
import utils
from net_utils import train_gacc, validate, test, EarlyStopping, deactivate_batchnorm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    selected_device = config['device']
    device = torch.device(selected_device if torch.cuda.is_available() else "cpu")
    
    if config["neptune"]:
        run = neptune.init_run(project="ProjektMMG/MCDO")
        run["sys/group_tags"].add(["no-BN"])
        run["sys/group_tags"].add(["ImageNet-norm"])
        run["sys/group_tags"].add(["pre-softmax-do"])
        run["sys/group_tags"].add(["XAI-EVAL"])
        run["config"] = config
    else:
        run = None

    SEED = config['seed']
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

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

    dataloaders = utils.get_dataloaders(config)
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

    for epoch in range(1, config['training_plan']['parameters']['epochs'] + 1):
        train_gacc(model=model, dataloader=dataloaders['train'],
                   criterion=criterion, optimizer=optimizer, device=device, neptune_run=run, epoch=epoch,
                   accumulation_steps=config['training_plan']['parameters']['grad_acc_steps'])
        val_loss = validate(model=model, dataloader=dataloaders['val'],
                            criterion=criterion, device=device, neptune_run=run, epoch=epoch)
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break
    model_name = uuid.uuid4().hex
    model_name = os.path.join(config['model_path'], model_name)
    torch.save(early_stopping.get_best_model_state(), model_name)
    if run is not None:
        run["best_model_path"].log(model_name)
    model = MultiHeadGatedAttentionMIL(
        backbone=config['model'],
        feature_dropout=config['feature_dropout'],
        attention_dropout=config['attention_dropout'],
        shared_attention=config['shared_att']
        )
    model.apply(deactivate_batchnorm)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    test(model, dataloaders['test'], device, run)
    if run is not None:
        run.stop()
# %%
