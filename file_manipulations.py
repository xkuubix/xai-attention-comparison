from pydicom import dcmread
import pandas as pd
import os
import numpy as np
import json
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

def read_metadata(file_path):
    return pd.read_csv(file_path)


def read_excel(file_path, sheet_name=1):
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])
    data.columns = [col[0] if "Unnamed" in str(col[1]) else f"{col[0]}_{col[1]}" for col in data.columns]
    return data


def get_metadata(path):
    dcm_count = 0
    dcm_mlo_count = 0
    all_metadata = {}

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.dcm'):
                dcm_count += 1
                ds = dcmread(os.path.join(dirpath, filename), stop_before_pixels=True)
                patient_id = ds.get('PatientID')
                metadata = {
                    'Laterality': ds.get('ImageLaterality', None),
                    'View': ds.get('ViewCodeSequence')[0].get('CodeMeaning') if ds.get('ViewCodeSequence') else None,
                    'FilePath': os.path.join(dirpath, filename)
                }
                if patient_id not in all_metadata:
                    all_metadata[patient_id] = {"Laterality+View": {"LMLO": [], "RMLO": []}}
                if metadata['View'] != 'medio-lateral oblique':
                    continue # TOMPEI-CMMD only has medio-lateral oblique annotations
                dcm_mlo_count += 1
                all_metadata[patient_id]['Laterality+View'][metadata['Laterality']+'MLO'].append(metadata['FilePath'])
    print(f"Total DICOM files found: {dcm_count} with {dcm_mlo_count} MLO images")
    return all_metadata


def apply_exclusion_criteria(df_excel, all_metadata):
    for _, row in df_excel.iterrows():
        patient_id = row['ID']
        exclusion = row['classification_Exclusion reasons']
        side = row['LeftRight']

        if pd.notna(exclusion) and patient_id in all_metadata:
            views = all_metadata[patient_id].get('Laterality+View', {})
            
            if side == 'L' and 'LMLO' in views:
                views['LMLO'] = ()
            elif side == 'R' and 'RMLO' in views:
                views['RMLO'] = ()
    totalLMLO = 0
    totalRMLO = 0
    keys_to_delete = []
    for key in all_metadata.keys():
        L = len(all_metadata[key]['Laterality+View']['LMLO'])
        R = len(all_metadata[key]['Laterality+View']['RMLO'])
        totalLMLO += L
        totalRMLO += R
        if L == 0 and R == 0:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del all_metadata[key]
    print(f"Total DICOM files left after exclusion: {totalLMLO} LMLO, {totalRMLO} RMLO ({totalLMLO + totalRMLO} total)")


def map_classification(classification):
    cls = str(classification).lower()
    if cls in ['benign', 'normal']:
        return 'Negative'
    elif cls == 'malignant':
        return 'Positive'
    else: # Invisible or Exclusion
        return 'Unknown'
    

def merge_dataframes(df_images, df_annotations, df_excel):
    
    df_images_long = df_images.melt(
        id_vars='ID', 
        value_vars=['LMLO', 'RMLO'],
        var_name='LeftRight',
        value_name='ImagePath'
    )
    df_images_long = df_images_long[~df_images_long['ImagePath'].isin([[], ()])]
    df_images_long['LeftRight'] = df_images_long['LeftRight'].map({'LMLO': 'L', 'RMLO': 'R'})
    df_full = df_excel.merge(
        df_images_long,
        left_on=['ID', 'LeftRight'],
        right_on=['ID', 'LeftRight'],
        how='left'
    )

    df_positive = df_full[df_full['classificationMapped'] == 'Positive'].merge(
        df_annotations,
        on=['ID', 'LeftRight'],
        how='left'
    )

    df_full = df_full.merge(
        df_positive[['ID', 'LeftRight', 'AnnotPath']],
        on=['ID', 'LeftRight'],
        how='left'
    )

    df_full = df_full.dropna(subset=['ImagePath'])
    return df_full


def generate_filled_masks(df,
                          output_dir='output/masks',
                          if_save=True,
                          if_show=False,
                          df_save_path='output/df_with_masks.pkl'):

    # BINARY CASE - cancer vs. no cancer masks
    
    os.makedirs(output_dir, exist_ok=True)
    mask_paths = []

    for idx, row in df.iterrows():
        print(f"Processing image {idx + 1}/{len(df)}")

        patient_id = row['ID']
        laterality = row.get('LeftRight', 'Unknown')
        classification = row.get('classificationMapped', 'Unknown')
        image_path = row.get('ImagePath')[0]
        annotation_path = row.get('AnnotPath', None)

        ds = pydicom.dcmread(image_path)
        image_array = ds.pixel_array
        height, width = image_array.shape
        mask = np.zeros((height, width), dtype=np.bool)

        if pd.notna(annotation_path):
            with open(annotation_path, 'r') as f:
                annotation_data = json.load(f)
            for ann in annotation_data:
                points = ann.get("cgPoints", [])
                if not points:
                    continue
                polygon = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon], color=1)

        if if_save:
            filename = f'{patient_id}_{laterality}_{classification}_mask.png'
            save_path = os.path.join(output_dir, filename)
            Image.fromarray(mask).save(save_path)
            print(f"Saved mask to {save_path}")
            mask_paths.append(save_path)
        else:
            mask_paths.append(None)

        if if_show:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image_array, cmap='gray')
            ax.imshow(mask, alpha=0.4, cmap='Reds')
            plt.title(f'ID: {patient_id} | dx: {classification} | {laterality}')
            plt.axis('off')
            plt.show()
            plt.close(fig)

    df['MaskPath'] = mask_paths
    if if_save:
        os.makedirs(os.path.dirname(df_save_path), exist_ok=True)
        df.to_pickle(df_save_path)
    return df
