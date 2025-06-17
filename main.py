#%%
import matplotlib.pyplot as plt
from file_manipulations import *

if __name__ == "__main__":
    file_path = '../metadata.csv'
    excel_path = '../TOMPEI-CMMD_clinical_data_v01_20250121.xlsx'
    path = '/users/project1/pt01190/TOMPEI-CMMD/CMMD-original/'

    df_excel = read_excel(excel_path)
    all_metadata = make_metadata(path)
    apply_exclusion_criteria(df_excel, all_metadata)

#%%
# TODO visualize annotations
    for patient_id, studies in all_metadata.items():
        for study, images in studies.items():
            for img_metadata in images:
                ds = dcmread(img_metadata['FilePath'])
                plt.imshow(ds.pixel_array, cmap='gray')
                plt.title(f"PatientID: {patient_id}, View: {img_metadata['View']}, Laterality: {img_metadata['Laterality']}")
                plt.show()



#%%