# %%
import quantus
import yaml, logging
import os, random, numpy as np
if 'code' not in os.getcwd():
    os.chdir('code/')
import torch    
import neptune
from models import *
import utils
from quantus_utils import *
from net_utils import deactivate_batchnorm
import matplotlib.pyplot as plt
plt.rc('font', family='Nimbus Roman')
plt.rcParams['font.size'] = 16


if __name__ == "__main__":
    if not os.getcwd().endswith('code'):
        os.chdir('code')
    logging.basicConfig(level=logging.INFO)
    print("Quantus version:", quantus.__version__)
    parser = utils.get_args_parser()
    args, unknown = parser.parse_known_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project = neptune.init_project(project="ProjektMMG/MCDO")

    ranges = [
        # (552, 558), #mh tompei
        # (560, 565), #sh tompei
        # (709, 719), # ds+clam tompei
        (754, 779), # eci (4x5)
        ]
    
    runs_table_df = project.fetch_runs_table(
        # id=[f"MCDO-{id}" for id in range(734, 740)],
        id=[f"MCDO-{i}" for start, end in ranges for i in range(start, end)],
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
        
        if run['model/architecture'] == "GatedAttentionMIL":
            model = GatedAttentionMIL(
                backbone=run['config/model'],
                feature_dropout=run['config/feature_dropout'],
                attention_dropout=run['config/attention_dropout'],
                config=model_config
                )
        elif run['model/architecture'] == "MultiHeadGatedAttentionMIL":
            model = MultiHeadGatedAttentionMIL(
                backbone=run['config/model'],
                feature_dropout=run['config/feature_dropout'],
                attention_dropout=run['config/attention_dropout'],
                shared_attention=run['config/shared_att'],
                config=model_config
                )
        elif run['model/architecture'] == "DSMIL":
            model = DSMIL(
                 num_classes=1,
                 backbone=run['config/model'],
                 dropout_v=run['config/feature_dropout'],
                 config=model_config
                 )
        elif run['model/architecture'] == "CLAM":
            model = CLAM(gate=True, size_arg="small", dropout=0.1, k_sample=8, n_classes=2,
                         instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=512,
                         backbone='r18', pretrained=True, output_class=1, config=None)
        
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
        model_name = run['best_model_path']
        
        fold = utils.get_fold_number(model_name) - 1
        if fold != 0:
            print(f"Skipping fold {fold+1}")
            continue
        dataloaders = utils.get_fold_dataloaders(data_config, fold)
        test_loader = dataloaders['test']

        print(f"Loading model from: {model_name}")
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        # test(model, dataloaders['test'], device, None)
        # mc_test(model, dataloaders['test'], device, N=100, fold_idx=None)

        test_loader = dataloaders['test']

        os.chdir('/users/project1/pt01190/TOMPEI-CMMD/code')
        # path = "../results-ieee/attentions-cv-plus-tompei-tompei/"
        # path = "../results-ieee/attentions-cv-plus-tompei-eci/"
        # path = "../results-ieee/attentions-cv-plus-eci-eci/"
        path = "../results-ieee/attentions-cv-plus-eci-tompei/"
        if not os.path.exists(path):
            print(f"\nCreating attention directory at {path}")
            os.mkdir(path)

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
                if label.item() == 0:
                    continue
                model.reconstruct_attention = True
                model.eval()
                
                y, extra = model(image)
                if model.__class__.__name__ == "DSMIL":
                    A0 = extra[0].cpu().numpy()
                    A1 = None
                elif model.__class__.__name__ == "MultiHeadGatedAttentionMIL":
                    A0 = extra[0, 0, 0, :, :].cpu().numpy()  # Choose attention map of head 0
                    A1 = extra[0, 1, 0, :, :].cpu().numpy()  # Choose attention map of head 1
                else:
                    A0 = extra.cpu().numpy()
                    A1 = None
              
                # === Original Image ===
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                ax1.imshow(image[0, :, :, :].permute(1, 2, 0).cpu().numpy())
                ax1.axis('off')
                fig1.tight_layout()
                fig1.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_original.png", dpi=300)
                # fig.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_original.pdf", format='pdf')
                plt.close(fig1)

                # === Mask (if available) ===
                fig3, ax3 = plt.subplots(figsize=(6, 6))
                ax3.imshow(mask[0, :, :].cpu().numpy(), alpha=0.5, cmap='gray')
                ax3.axis('off')
                fig3.tight_layout()
                fig3.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_mask.png", dpi=300)
                # fig.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_mask.pdf", format='pdf')
                plt.close(fig3)

                # === Attention Map ===
                print(f"Attention shape: {A0.shape}")
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.imshow(A0.squeeze(), cmap='jet')
                ax2.axis('off')
                fig2.tight_layout()
                fig2.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_attention0.png", dpi=300)
                # fig2.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_attention.pdf", format='pdf')
                plt.close(fig2)

                if A1 is not None:
                    fig4, ax4 = plt.subplots(figsize=(6, 6))
                    ax4.imshow(A1.squeeze(), cmap='jet')
                    ax4.axis('off')
                    fig4.tight_layout()
                    fig4.savefig(f"{path}/{run['sys/id']}/{batch['metadata']['ID'][0]}_attention1.png", dpi=300)
                    plt.close(fig4)