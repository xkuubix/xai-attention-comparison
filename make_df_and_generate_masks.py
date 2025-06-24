#%%
from file_manipulations import *
import os


if __name__ == "__main__":
    file_path = '../metadata.csv'
    excel_path = '../TOMPEI-CMMD_clinical_data_v01_20250121.xlsx'
    path = '/users/project1/pt01190/TOMPEI-CMMD/CMMD-original/'
    annotation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../annotations'))
    annotation_files = os.listdir(annotation_dir)

    df_excel = read_excel(excel_path)
    some_metadata = get_metadata(path)
    apply_exclusion_criteria(df_excel, some_metadata)
    df_excel = df_excel.iloc[:, :4]
    df_excel = df_excel.rename(columns={'classification': 'Classification'})

    df_excel['ClassificationMapped'] = df_excel['Classification'].apply(map_classification)
    df_excel = df_excel[df_excel['ClassificationMapped'] != 'Unknown']

    id_view_lat = [item[:13] for item in annotation_files] # Extracting {ID}_{View}_{Laterality} from filenames
    all_annotations = []
    for item, file in zip(id_view_lat, annotation_files):
        temp = item.split('_')
        annot_dict = {"ID": temp[0], "View": temp[1], "LeftRight": temp[2], "AnnotPath": os.path.join(annotation_dir, file)}
        all_annotations.append(annot_dict)

    df_annotations = pd.DataFrame(all_annotations)
    df_annotations = df_annotations.sort_values(by='ID').reset_index(drop=True)

    records = []
    for patient_id, studies in some_metadata.items():
        for k, v in studies.items():
            record = {"ID": patient_id, "RMLO": None, "LMLO": None}
            record["LMLO"] = studies[k].get('LMLO', [])
            record["RMLO"] = studies[k].get('RMLO', [])
            records.append(record)
    df_images = pd.DataFrame(records).sort_values(by='ID').reset_index(drop=True)
    
    # several images contain both benign and malignant lesion, we will remove them
    simul_bg_mg = ["D1-0397", "D1-0869", "D2-0033", "D2-0116", "D2-0133", "D2-0185", "D2-0637"]

    df_images = df_images[~df_images['ID'].isin(simul_bg_mg)]
    df_excel = df_excel[~df_excel['ID'].isin(simul_bg_mg)]
    df_annotations = df_annotations[~df_annotations['ID'].isin(simul_bg_mg)]

    df_full = merge_dataframes(df_images, df_annotations, df_excel)

    wrong_laterality_ids = ['D2-0224', 'D2-0229', 'D2-0642'] # IDs with wrong laterality in annotations or DICOM files (checked manually)
    
    for wrong_id in wrong_laterality_ids:
        subset = df_full[df_full['ID'] == wrong_id]
        if len(subset) != 2:
            print(f"Skipping {wrong_id}: expected 2 rows, found {len(subset)}")
            continue

        idx1, idx2 = subset.index
        df_full.loc[idx1, 'LeftRight'], df_full.loc[idx2, 'LeftRight'] = df_full.loc[idx2, 'LeftRight'], df_full.loc[idx1, 'LeftRight']
        df_full.loc[idx1, 'AnnotPath'], df_full.loc[idx2, 'AnnotPath'] = df_full.loc[idx2, 'AnnotPath'], df_full.loc[idx1, 'AnnotPath']
    
    df_full["ImagePath"] = df_full["ImagePath"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)    

    df = generate_filled_masks(df_full,
                               output_dir='../masks',
                               if_save=True,
                               if_show=False,
                               df_save_path='../df_with_masks.pkl')

#%%
