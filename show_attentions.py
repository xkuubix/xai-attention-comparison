# %%
import quantus
import yaml, logging
import os, random, numpy as np
if 'code' not in os.getcwd():
    os.chdir('code/')
import torch    
import neptune
from models import GatedAttentionMIL, MultiHeadGatedAttentionMIL
import utils
from quantus_utils import *
from net_utils import deactivate_batchnorm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Quantus version:", quantus.__version__)

    project = neptune.init_project(project="ProjektMMG/MCDO")

    runs_table_df = project.fetch_runs_table(
        id=[f"MCDO-{id}" for id in range(490, 500)],
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

        test_loader = dataloaders['test']

        if not os.path.exists(f"attentions/{run['sys/id']}"):
            os.mkdir(f"attentions/{run['sys/id']}")
        else:
            print(f"\nAttention directory for run {run['sys/id']} already exists. Skipping attention visualization.")
            continue

        with torch.no_grad():
            for batch in test_loader:
                image = batch['image']
                label = batch['target']['label']
                mask = batch['target'].get('mask', None)
                # if label.item() == 0:
                #     continue
                model.reconstruct_attention = True
                model.eval()
                y, A = model(image)
                
                if model.__class__.__name__ == 'MultiHeadGatedAttentionMIL':
                    print(A.shape)
                    A = A[0, 1, 0, :, :].cpu().numpy()
                elif model.__class__.__name__ == 'GatedAttentionMIL':
                    A = A[0, 0, 0, :, :].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(16, 6))

                # Original image
                axes[0].imshow(image[0, :, :, :].permute(1, 2, 0).cpu().numpy())
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                # Attention
                print(f"Attention shape: {A.shape}")
                axes[1].imshow(A, cmap='jet')
                axes[1].set_title("Attention Map")
                axes[1].axis('off')

                # Mask (if available)
                if mask is not None:
                    axes[2].imshow(mask[0, :, :].cpu().numpy(), alpha=0.5, cmap='gray')
                axes[2].set_title("Mask")
                axes[2].axis('off')

                label_str = 'Negative' if label.item() == 0 else 'Positive'

                pos_prob = torch.softmax(y, dim=1)[0, 1].item() if SOFTMAX else torch.sigmoid(y).item()

                plt.suptitle(f"ID: {batch['metadata']['ID'][0]}    |   Ground Truth Label: {label_str}    |    Positive Prob.: {pos_prob:.2f}",
                             fontsize=16)
                plt.tight_layout()
                plt.show()
                plt.savefig(f"attentions/{run['sys/id']}/{batch['metadata']['ID'][0]}_attention.png")
                plt.close(fig)


# %%
