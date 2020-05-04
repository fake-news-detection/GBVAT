import pandas as pd

df = pd.read_csv("indian_liver_patient.csv")

df.nunique()
df.head()


import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),cmap='magma',annot=True)
plt.show()


corr = df.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);