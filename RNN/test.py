import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("indian_liver_patient.csv")

# # instantiate the figure
# fig = plt.figure(figsize = (10, 5))
# gs = fig.add_gridspec(5, 5)
# ax1 = fig.add_subplot(gs[:4, :-1])
#
# x, y = df['Alkaline_Phosphotase'], df['Age']
#
# ax1.scatter(x, y, c = df.Dataset.astype('category').cat.codes)
# ax1.set_xlabel("Dist")
# ax1.set_ylabel("Hwy")
# ax1.set_title("Scatter plot with marginal histograms")
#
# # Removes the boundaries
# ax1.spines["right"].set_color("None")
# ax1.spines["top"].set_color("None")
#
#
# ax2 = fig.add_subplot(gs[4:, :-1])
# ax2.hist(x, 40, orientation = 'vertical', color = "blue")
# ax2.axison = False
#
# ax3 = fig.add_subplot(gs[:4, -1])
# ax3.hist(y, 40, orientation = "horizontal", color = "blue")
# ax3.axison = False
#
# plt.show()

# instanciate the figure
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot()

# plot using matplotlib
ax.imshow(df.corr(), cmap = 'viridis', interpolation = 'nearest')
# set the title for the figure
ax.set_title("Heatmap using matplotlib")

plt.show()