#%%
from pydicom import dcmread
import pandas as pd
import os

def read_metadata(file_path):
    return pd.read_csv(file_path)

def read_excel(file_path, sheet_name=1):
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])
    data.columns = [col[0] if "Unnamed" in str(col[1]) else f"{col[0]}_{col[1]}" for col in data.columns]
    return data


if __name__ == "__main__":
    file_path = 'metadata.csv'
    excel_path = 'TOMPEI-CMMD_clinical_data_v01_20250121.xlsx'
    df = read_metadata(file_path)
    df_excel = read_excel(excel_path)
    path = '/users/project1/pt01190/TOMPEI-CMMD/CMMD/D2-0252/07-18-2011-NA-NA-21941/1.000000-NA-99533'
    
    for file in os.listdir(path):
        print(f"Found file: {file}")
        if file.endswith('.dcm'):
            full_path = os.path.join(path, file)
            ds = dcmread(full_path)
            metadata = {
                'PatientID': ds.get('PatientID'),
                'Laterality': ds.get('ImageLaterality', None),
                'View': ds.get('ViewCodeSequence')[0].get('CodeMeaning'),
            }
            print(f"Processing file: {metadata}")



#%%