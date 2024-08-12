import torch
import matplotlib.pyplot as plt
from koipond.nn.data import KoiPond

if __name__ == '__main__':
    koi_pond = KoiPond('data')
    test_len = 100
    koi_test_pond, _  = torch.utils.data.random_split(koi_pond, [test_len, len(koi_pond)-test_len])
    test_loader = torch.utils.data.DataLoader(koi_test_pond, batch_size=1, shuffle=False)
    with torch.no_grad():
        count = 1
        for data in test_loader:
            curves, labels = data
            fig, ax = plt.subplots()
            ax.plot(curves.flatten().numpy(), 'k.', markersize=2)
            ax.set_title(f'Label: {labels.flatten().numpy()[0]}')
            fig.savefig(f'img/curve{count}')
            plt.close(fig)
            count+=1
    print('Done')
