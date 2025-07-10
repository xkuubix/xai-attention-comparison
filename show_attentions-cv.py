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
        id=[f"MCDO-{id}" for id in range(552, 565)],
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
        model_name = run['best_model_path']
        
        fold = utils.get_fold_number(model_name) - 1
        dataloaders = utils.get_fold_dataloaders(data_config, fold)
        test_loader = dataloaders['test']

        print(f"Loading model from: {model_name}")
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        # test(model, dataloaders['test'], device, None)
        # mc_test(model, dataloaders['test'], device, N=100, fold_idx=None)

        test_loader = dataloaders['test']

        os.chdir('/users/project1/pt01190/TOMPEI-CMMD/code')
        path = "attentions-cv-sep"
        if not os.path.exists(path):
            print(f"\nCreating attention directory at {path}")
            os.mkdir(path)
        # path = "attentions"

        if not os.path.exists(f"{path}/{run['sys/id']}"):
            os.mkdir(f"{path}/{run['sys/id']}")
            print(f"\nCreating attention directory for run {run['sys/id']} at {path}/{run['sys/id']}")
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
                
                if 'mcdo' in path:
                    for module in model.modules():
                        if isinstance(module, torch.nn.Dropout):
                            module.p = 0.75
                            print(f"Setting dropout probability to {module.p} for MC-Dropout inference.")
                    y, A = model.mc_inference(image, N=500)
                else:
                    y, A = model(image)
                # print(A.shape) torch.Size([1, 2, 1, 2294, 1914])
                if model.__class__.__name__ == 'MultiHeadGatedAttentionMIL':
                    if 'mcdo' in path:
                        # y shape: [N, 1, 2] — N = MC samples
                        y_activated = torch.softmax(y, dim=2)
                        y_avg = y_activated.mean(dim=0).squeeze(0)  # Average MC samples → shape: [2]
                    else:
                        y_avg = torch.softmax(y, dim=1).squeeze(0)  # shape: [2]
                    
                    pos_prob = y_avg[1].item()  # Class 1 probability
                    # A = A.mean(dim=0, keepdim=True)  # [N, ..., H, W] → average attention maps over MC samples
                    A = A[0, 1, 0, :, :].cpu().numpy()  # Choose attention map of head 1

                elif model.__class__.__name__ == 'GatedAttentionMIL':
                    if 'mcdo' in path:
                        # y shape: [N, 1, 1, 1]
                        y_activated = torch.sigmoid(y)  # Apply sigmoid to each MC sample
                        y_avg = y_activated.mean(dim=0).squeeze()  # Average over N, squeeze → scalar
                    else:
                        y_avg = torch.sigmoid(y).squeeze()

                    pos_prob = y_avg.item()
                    # A = A.mean(dim=0, keepdim=True)  # Average MC attention maps
                    A = A[0, 0, 0, :, :].cpu().numpy()

                # === Original Image ===
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                ax1.imshow(image[0, :, :, :].permute(1, 2, 0).cpu().numpy())
                ax1.set_title("Original Image")
                ax1.axis('off')
                fig1.tight_layout()
                fig1.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_original.png", dpi=300)
                fig1.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_original.pdf", format='pdf')
                plt.close(fig1)

                # === Attention Map ===
                print(f"Attention shape: {A.shape}")
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.imshow(A, cmap='jet')
                ax2.set_title("Attention Map")
                ax2.axis('off')
                fig2.tight_layout()
                fig2.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_attention.png", dpi=300)
                fig2.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_attention.pdf", format='pdf')
                plt.close(fig2)

                # === Mask (if available) ===
                if mask is not None:
                    fig3, ax3 = plt.subplots(figsize=(6, 6))
                    ax3.imshow(mask[0, :, :].cpu().numpy(), alpha=0.5, cmap='gray')
                    ax3.set_title("Mask")
                    ax3.axis('off')
                    fig3.tight_layout()
                    fig3.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_mask.png", dpi=300)
                    fig3.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_mask.pdf", format='pdf')
                    plt.close(fig3)
