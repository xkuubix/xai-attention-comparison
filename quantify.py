# %%
import quantus
import yaml, logging
import os, random, numpy as np
import torch    
import torch.nn as nn
import neptune
from models import GatedAttentionMIL, MultiHeadGatedAttentionMIL
import utils
from net_utils import deactivate_batchnorm, test, mc_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Quantus version:", quantus.__version__)

    project = neptune.init_project(project="ProjektMMG/MCDO")

    runs_table_df = project.fetch_runs_table(
        id=[f"MCDO-{id}" for id in range(490, 491)],
        owner="jakub-buler",
        state="inactive",
        trashed=False,
        ).to_pandas()

    for _, run in runs_table_df.iterrows():
        selected_device = run['config/device']
        device = torch.device(selected_device if torch.cuda.is_available() else "cpu")
        SEED = run['config/seed']
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        torch.set_default_dtype(torch.float32)

        model_config = {"data": {
            "patch_size": run['config/data/patch_size'],
            "overlap": run['config/data/overlap'],
            "bag_size": run['config/data/bag_size'],
            "empty_threshold": run['config/data/empty_threshold']},
            }
        if run['model/architecture'] == 'GatedAttentionMIL':
            model = GatedAttentionMIL(
                backbone=run['config/model'],
                feature_dropout=run['config/feature_dropout'],
                attention_dropout=run['config/attention_dropout'],
                config=model_config
                )
            SOFTMAX = False
        elif run['model/architecture'] == 'MultiHeadGatedAttentionMIL':
            model = MultiHeadGatedAttentionMIL(
                backbone=run['config/model'],
                feature_dropout=run['config/feature_dropout'],
                attention_dropout=run['config/attention_dropout'],
                shared_attention=run['config/shared_att'],
                config=model_config
                )
            SOFTMAX = True

        print("--"*30)
        print(f"Run ID: {run['sys/id']}")
        print("--"*30)
        print(yaml.dump(model_config, sort_keys=False, default_flow_style=False))
        print(f"\n\nModel architecture:\n\n | {str(model.__class__.__name__)} |\n\n")
        model.apply(deactivate_batchnorm)

        data_config = {
            "data": {
            "metadata_path": run['config/data/metadata_path'],
            "fraction_train_rest": run['config/data/fraction_train_rest'],
            "fraction_val_test": run['config/data/fraction_val_test']
            },
            "training_plan": {
            "criterion": run['config/training_plan/criterion'],
            "parameters": {
                "batch_size": run['config/training_plan/parameters/batch_size'],
                "num_workers": run['config/training_plan/parameters/num_workers']
            }
            },
            "seed": SEED,
            "device": selected_device
        }
        dataloaders = utils.get_dataloaders(data_config)

        model_name = run['best_model_path']
        print(f"Loading model from: {model_name}")
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        # test(model, dataloaders['test'], device, None)
        # mc_test(model, dataloaders['test'], device, N=100, fold_idx=None)



    # SINGLE RUN FOR NOW

    test_loader = dataloaders['test']

    from quantus_utils import *

    # a = evaluate_sparseness(model, test_loader)
   
    # b = evaluate_selectivity(model, test_loader, softmax=SOFTMAX)
   
    c = evaluate_local_lipschitz_estimate(model, test_loader)

    # b = evaluate_faithfulness_correlation(model, test_loader)

    # c = evaluate_relevance_rank_accuracy(model, test_loader)
    


# %%
