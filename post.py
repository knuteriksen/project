import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathmanager import get_results_path

sns.set_theme()

vals = pd.read_csv(os.path.join(get_results_path(), "easy_plot.csv"))
ax = sns.lineplot(data=vals)
ax.set(ylim=(0, 20))
plt.show()
