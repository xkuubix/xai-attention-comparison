from pydicom import dcmread
import pandas as pd
import os


def read_metadata(file_path):
    return pd.read_csv(file_path)


def read_excel(file_path, sheet_name=1):
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])
    data.columns = [col[0] if "Unnamed" in str(col[1]) else f"{col[0]}_{col[1]}" for col in data.columns]
    return data


def make_metadata(path):
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
    for key in all_metadata.keys():
        totalLMLO += len(all_metadata[key]['Laterality+View']['LMLO'])
        totalRMLO += len(all_metadata[key]['Laterality+View']['RMLO'])
    print(f"Total DICOM files left after exclusion: {totalLMLO} LMLO, {totalRMLO} RMLO ({totalLMLO + totalRMLO} total)")
