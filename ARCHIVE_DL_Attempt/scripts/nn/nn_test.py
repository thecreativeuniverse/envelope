import torch
from koipond.nn.koinet import KoiNet
from koipond.nn.data import KoiPond
from koipond.util.constants import INPUT_SIZE
import argparse, os


def load_and_test(file="models/koi_net.pt", batch_size=64, test_loader=None):
    koi_net = torch.load(file)
    print(f"Testing file {file}")
    # TODO: move this list to Somewhere Else.
    # 28th Jan (train 1)::v1:  69% accuracy at 2 epochs, ~45k data files
    # 29th Jan (train 2)::v1:  67% accuracy at 4 epochs, ~43k data files
    # 29th Jan (train 3)::v2:  67% accuracy at 10 epochs, 696 batches of size 64.
    # koi_net = KoiNet(INPUT_SIZE)# untrained one: 30% accuracy
    # 29th Jan (train 2): 
    if test_loader is None:
        koi_pond = KoiPond('data')
        test_len = round(len(koi_pond)/5)
        koi_test_pond, _  = torch.utils.data.random_split(koi_pond, [test_len, len(koi_pond)-test_len])
        test_loader = torch.utils.data.DataLoader(koi_test_pond, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            curves, labels = data
            # calculate outputs by running images through the network
            outputs = koi_net(curves)
            predicted = torch.round(outputs).view(outputs.size(0))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(outputs)
            print(predicted)
            print(labels)
            print(correct)
            print(total)
            break
    print(f'Accuracy of the network on the {len(test_loader) * batch_size} test data: {100 * correct // total} %')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test KoiNet.")
    parser.add_argument('-u', '--uuid', help='UUID of model to test.')
    parser.add_argument('-f', '--file', help='Specific file to test.')
    parser.add_argument('-b', '--batchsize', help='Batch Size', default=64, type=int)

    args = parser.parse_args()

    if args.uuid is not None:
        koi_pond = KoiPond('data')
        test_len = round(len(koi_pond)/5)
        koi_test_pond, _  = torch.utils.data.random_split(koi_pond, [test_len, len(koi_pond) - test_len])
        test_loader = torch.utils.data.DataLoader(koi_test_pond, batch_size=args.batchsize, shuffle=False)
        for fname in os.listdir('models/'):
            if args.uuid in fname:
                load_and_test(file=f"models/{fname}", batch_size=args.batchsize, test_loader=test_loader)
    elif args.file is not None:
        load_and_test(file=args.file, batch_size=args.batchsize)
