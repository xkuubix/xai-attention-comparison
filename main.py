# %%
import pickle
from TompeiDataset import TompeiDataset
from torchvision import transforms as T

df = pickle.load(open('/users/project1/pt01190/TOMPEI-CMMD/df_with_masks.pkl', 'rb'))

# TODO DATA SPLIT (by ID)
# TODO TRANSFORMATIONS

train_dataset = TompeiDataset(df, transform=T.ToTensor())




# %%
