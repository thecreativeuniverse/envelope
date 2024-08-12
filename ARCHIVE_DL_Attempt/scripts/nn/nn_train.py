from koipond.nn.koinet import KoiNet
from koipond.nn.data import KoiPond
from koipond.util.constants import INPUT_SIZE
import torch
import uuid, argparse, pandas
import http.client as http
import urllib.parse
import traceback, sys

def validate(koi_net, koi_test_loader, criterion):
    print("Validating data set.")
    total = 0
    correct = 0

    running_loss = 0.0

    with torch.no_grad():
        for data in koi_test_loader:
            curves, labels = data
            # calculate outputs by running images through the network
            outputs = koi_net(curves)
            predicted = torch.round(outputs.flatten())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs.flatten(), labels)
            running_loss += loss.item()

    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the {len(koi_test_loader) * batch_size} test data: {accuracy}% with loss {running_loss/total}')
    return accuracy, running_loss/total

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train KoiNet model.")
        parser.add_argument('-f', '--file', help='Relative path to file where model is stored (optional)')
        parser.add_argument('--notify', '-n', action='store_true', help='Notify to phone when done?')
        parser.add_argument('-e', '--epoch', help='Num epochs to run', default=10, type=int)
        parser.add_argument('-b','--batchsize', help='Batch size', default=128, type=int)
        args = parser.parse_args()

        batch_size = args.batchsize
        epoch_size = args.epoch

        koi_pond = KoiPond('data')
        train_len = round(4 * len(koi_pond) / 5)
        train_pond, test_pond = torch.utils.data.random_split(koi_pond, [train_len, len(koi_pond) - train_len])
        koi_train_loader = torch.utils.data.DataLoader(train_pond, batch_size=batch_size, shuffle=True, drop_last=True)
        koi_test_loader = torch.utils.data.DataLoader(test_pond, batch_size=batch_size, shuffle=True, drop_last=True)

        if args.file is not None:
            try:
                print(f"Loading KoiNet from {args.file}...")
                koi_net = torch.load(args.file)
            except:
                print(f"Could not load koi net {args.file}. Creating blank KoiNet.")
                koi_net = KoiNet(INPUT_SIZE)
        else:
            print("Initialising blank KoiNet.")
            koi_net = KoiNet(INPUT_SIZE)

        criterion = torch.nn.BCELoss()
        # optimizer = torch.optim.SGD(koi_net.parameters(), lr=0.001, momentum=0.5)
        optimizer = torch.optim.Adam(koi_net.parameters(), lr=0.001)
        train_id = uuid.uuid4()

        validation_df = pandas.DataFrame({'epoch':[],'accuracy':[],'loss':[]})

        print(f"Running training with runtime identifier {train_id}, using {len(koi_train_loader)} batches of size {batch_size}.")

        accuracy, loss = validate(koi_net, koi_test_loader, criterion)
        validation_df = pandas.concat((validation_df, pandas.DataFrame({'epoch':[0],'accuracy':[accuracy], 'loss':[loss]})))
        validation_df.to_csv(path_or_buf=f'models/{train_id}.csv')

        # basic training algo from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for epoch in range(epoch_size):
            running_loss = 0.0
            running_total = 0

            for i, data in enumerate(koi_train_loader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = koi_net(inputs)
                    outputs = outputs.flatten()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    running_total += 1
                    if i % 100 == 0:    # print every 100 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / running_total:.3f}')
                        running_loss = 0.0
                        running_total = 0
            file_name = f"models/koinet_{train_id}_epoch{epoch+1}_total.pt"
            print(f"Saving torch file to {file_name}. Completed {i} batches in this epoch.")
            torch.save(koi_net, f=file_name)
            accuracy,loss = validate(koi_net, koi_test_loader, criterion)
            validation_df = pandas.concat((validation_df, pandas.DataFrame({'epoch':[epoch+1],'accuracy':[accuracy], 'loss':loss})))
            validation_df.to_csv(path_or_buf=f'models/{train_id}.csv')

        print("Finished Training.")

    except Exception as e:
        print(e)
        traceback.print_exception(*sys.exc_info()) 

        if args.notify:
            print("Sending notification to phone")
            # this is mainly a temp solution. pushover tokens stored outside of git project.
            with open("../keys/api_token") as app_file:
                app_token = app_file.readline().replace("\n", "")
            with open("../keys/user_token") as user_file:
                user_token = user_file.readline().replace("\n", "")

            conn = http.HTTPSConnection("api.pushover.net:443")
            conn.request("POST", "/1/messages.json",
                    urllib.parse.urlencode({
                        "token": app_token,
                        "user": user_token,
                        "title": "Google Cloud",
                        "message": f"Exception in training job: {str(e)}",
                    }), {"Content-type":"application/x-www-form-urlencoded"})
            res = conn.getresponse()
            print(res.read())
    else:
        if args.notify:
            print("Sending notification to phone")
            # this is mainly a temp solution. pushover tokens stored outside of git project.
            with open("../keys/api_token") as app_file:
                app_token = app_file.readline().replace("\n", "")
            with open("../keys/user_token") as user_file:
                user_token = user_file.readline().replace("\n", "")

            conn = http.HTTPSConnection("api.pushover.net:443")
            conn.request("POST", "/1/messages.json",
                    urllib.parse.urlencode({
                        "token": app_token,
                        "user": user_token,
                        "title": "Google Cloud",
                        "message": f"Training job finished with test accuracy of {accuracy}%.",
                    }), {"Content-type":"application/x-www-form-urlencoded"})
            res = conn.getresponse()
            print(res.read())
