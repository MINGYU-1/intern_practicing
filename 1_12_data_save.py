import pandas as pd
import numpy as np
import os

df = pd.read_csv("211210-DRM-total.csv")
metal_df = df.iloc[:,1:25] #24
supporting_df = df.iloc[:,25:70] #45
pretreatment_df = df.iloc[:,70:76] #6
reaction_df = df.iloc[:,76:] #9

## npy로 저장하기
save_dir = './data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save('./data/metal.npy',metal_df)
np.save('./data/support.npy',supporting_df)
np.save('./data/pretreatment.npy',pretreatment_df)
np.save('./data/reaction.npy',reaction_df)