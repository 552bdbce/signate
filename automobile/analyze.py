import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re

data = pd.read_csv('./data/train.tsv', sep='\t')
data = data.drop(columns=['id'])
data = data[data.horsepower != '?']
data = data.dropna()
data['displacement_log'] = np.log(data['displacement'])
data['weight_log'] = np.log(data['weight'])

# get car manufacturer from 'car name'
data['car name'] = data['car name'].str.replace(r'([^\s]+).*', r'\1', regex=True)

X = data[['cylinders', 'displacement_log', 'horsepower', 'weight_log', 'acceleration', 'model year', 'origin', 'car name']]
X = pd.get_dummies(X)  # 各説明変数のダミー変数化
y = data['mpg']

fig = plt.figure()

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

l1, l2, l3, l4, l5, l6, l7, l8 = 'cylinders', 'displacement_log', 'horsepower', 'weight_log', 'acceleration', \
                             'model year', 'origin', 'car name'

ax1.scatter(data['cylinders'], data['mpg'], label=l1, s=5)
ax2.scatter(data['displacement_log'], data['mpg'], label=l2, s=5)
ax3.scatter(data['horsepower'], data['mpg'], label=l3, s=5)
ax4.scatter(data['weight_log'], data['mpg'], label=l4, s=5)
ax5.scatter(data['acceleration'], data['mpg'], label=l5, s=5)
ax6.scatter(data['model year'], data['mpg'], label=l6, s=5)
ax7.scatter(data['origin'], data['mpg'], label=l7, s=5)
ax8.scatter(data['car name'], data['mpg'], label=l8, s=5)


ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
ax5.legend(loc='upper right')
ax6.legend(loc='upper right')
ax7.legend(loc='upper right')
ax8.legend(loc = 'upper right')
# ax9.legend(loc = 'upper right')

fig.tight_layout()
plt.show()
