import pandas, torch, os
import numpy as np
import matplotlib.pyplot as plt
from koipond.util.constants import DATA_SEPARATION

count = 0
df = pandas.read_csv('data/init/raw.csv')
for file in os.listdir('data'):
    if not file.endswith('.pt'):
        continue
    curve = torch.load(f'data/{file}')
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,len(curve)*DATA_SEPARATION,DATA_SEPARATION), curve.data, 'k.', markersize=2)
    curve_id = file.split('_')[0]
    label = df.loc[df['curve_id'] == int(curve_id)]['label'].tolist()[0]
    print(label)
    ax.set_title(f"Label:{label} curve:{file.replace('.pt','')}")
    fig.savefig(f"data/imgs/{file.replace('.pt','.png')}")
    plt.close(fig)
    count += 1
    if count > 100:
        break
