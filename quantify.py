# %%
import sys
sys.path.append('/users/project1/pt01190/TOMPEI-CMMD/code')
import quantus
import yaml, logging
import os, random, numpy as np
import torch    
import torch.nn as nn
import neptune
from models import *
import utils
from quantus_utils import *
from net_utils import deactivate_batchnorm, test, mc_test
import pickle



if __name__ == "__main__":
    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    logging.basicConfig(level=logging.INFO)
    print("Quantus version:", quantus.__version__)

    project = neptune.init_project(project="ProjektMMG/MCDO")

    runs_table_df = project.fetch_runs_table(
        id=[f"MCDO-{id}" for id in range(517, 565)],
        owner="jakub-buler",
        state="inactive",
        trashed=False,
        ).to_pandas()

    for _, run in runs_table_df.iloc[::-1].iterrows():
        selected_device = run['config/device']
        device = torch.device(selected_device if torch.cuda.is_available() else "cpu")
        SEED = run['config/seed']
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        utils.reset_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.set_default_dtype(torch.float32)

        model_config = {"data": {
            "patch_size": run['config/data/patch_size'],
            "overlap": run['config/data/overlap'],
            "bag_size": run['config/data/bag_size'],
            "empty_threshold": run['config/data/empty_threshold']},
            }

        if run['model/architecture'] == "GatedAttentionMIL":
            model = GatedAttentionMIL(
                backbone=run['config/model'],
                feature_dropout=run['config/feature_dropout'],
                attention_dropout=run['config/attention_dropout'],
                config=model_config
                )
            is_single = True

        elif run['model/architecture'] == "MultiHeadGatedAttentionMIL":
            model = MultiHeadGatedAttentionMIL(
                backbone=run['config/model'],
                feature_dropout=run['config/feature_dropout'],
                attention_dropout=run['config/attention_dropout'],
                shared_attention=run['config/shared_att'],
                config=model_config
                )
            is_single = False

        elif run['model/architecture'] == "DSMIL":
            model = DSMIL(
                num_classes=1,
                backbone=run['config/model'],
                dropout_v=run['config/feature_dropout'],
                config=model_config
                )
            is_single = True

        elif run['model/architecture'] == "CLAM":
            model = CLAM(gate=True, size_arg="small", dropout=0.1, k_sample=8, n_classes=2,
                         instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=512,
                         backbone='r18', pretrained=True, output_class=1, config=None)        
            is_single = False

        else:
            raise ValueError("Model type not supported")

        print("--"*30)
        print(f"Run ID: {run['sys/id']}")
        print("--"*30)
        print(yaml.dump(model_config, sort_keys=False, default_flow_style=False))
        print(f"\n\nModel architecture:\n\n | {str(model.__class__.__name__)} |\n\n")
        model.apply(deactivate_batchnorm)

        data_config = {
            "data": {
            "metadata_path": config['data']['metadata_path'],
            "fraction_train_rest": run['config/data/fraction_train_rest'],
            "fraction_val_test": run['config/data/fraction_val_test'],
            "cv_folds": run['config/data/cv_folds'],
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
        
        # dataloaders = utils.get_dataloaders(data_config)

        fold = utils.get_fold_number(run['best_model_path']) - 1
        dataloaders = utils.get_fold_dataloaders(data_config, fold)
        test_loader = dataloaders['test']
        
        model_name = run['best_model_path']
        print(f"Loading model from: {model_name}")
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        
        os.chdir('/users/project1/pt01190/TOMPEI-CMMD/code')
        # folder_path = "../results-ieee/cv_results-plus-tompei-tompei"
        # folder_path = "../results-ieee/cv_results-plus-tompei-eci"
        # folder_path = "../results-ieee/cv_results-plus-eci-eci"
        # folder_path = "../results-ieee/cv_results-plus-eci-tompei"
        # folder_path = "../results-ieee/cv_results-plus-tompei-tompei-mh-neg"
        # folder_path = "../results-ieee/cv_results-plus-tompei-eci-mh-neg"
        # folder_path = "../results-ieee/cv_results-plus-eci-eci-mh-neg"
        folder_path = "../results-ieee/cv_results-plus-eci-tompei-mh-neg"
        os.makedirs(folder_path, exist_ok=True)
       
        EVALUATIONS = [
            {"name": "sparseness", "fn": evaluate_sparseness},
            # {"name": "topkintersection", "fn": evaluate_topk_intersection},
            {"name": "relevance_rank_accuracy", "fn": evaluate_relevance_rank_accuracy},
            # {"name": "road", "fn": lambda m, d: evaluate_road(m, d, use_wrapper=is_single)},
            # {"name": "avg_sensitivity", "fn": evaluate_avg_sensitivity},
            # {"name": "mprt", "fn": evaluate_mprt},
            # {"name": "faithfulness_correlation", "fn": lambda m, d: evaluate_faithfulness_correlation(m, d, use_wrapper=is_single)},
        ]

        for eval_task in EVALUATIONS:
            pickle_path = f"{folder_path}/{eval_task['name']}_{run['sys/id']}.pkl"

            if os.path.exists(pickle_path):
                print(f"Skipping {eval_task['name']} for run {run['sys/id']} as results already exist.")
            else:
                print(f"Evaluating {eval_task['name']} for run {run['sys/id']}")
                utils.reset_seed(SEED)
                with open(pickle_path, 'wb') as f:
                    result = eval_task["fn"](model, test_loader)
                    pickle.dump(result, f)

# %%
