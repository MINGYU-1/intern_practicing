import pandas as pd
import numpy as np
import os

df = pd.read_csv("211210-DRM-total.csv")
metal_df = df.iloc[:,2:25] #23
nickel_df = df.iloc[:,9:10] #nickel만
supporting_df = df.iloc[:,25:70] #45
nickel_except_df = df.iloc[:,list(range(2, 9)) + list(range(10, 25))] # nickle제외
pretreatment_df = df.iloc[:,70:76] #6
reaction_df = df.iloc[:,76:] #9

## npy로 저장하기
save_dir = './data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save('./data/metal.npy',metal_df)
np.save('./data/nickel.npy',nickel_df)
np.save('./data/support.npy',supporting_df)
np.save('./data/pretreatment.npy',pretreatment_df)
np.save('./data/reaction.npy',reaction_df)
np.save('./data/nickel_except.npy',nickel_except_df)

