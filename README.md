# AIMI-lab2
Lab2 EEG classification

### Train
1. Download `main.py` `dataloader.py` and `models` file
2. Check your environment have python, GPU and Pytorch
3. Adjustment parameters for training
4. Start training.
```
python main.py
```

### Adjustment parameters
```
#儲存圖片的編號
num = 30  #images name number

#繪製training accuracy plot
plot_train_acc(train_acc_list, epochs)

#繪製training loss plot
plot_train_loss(train_loss_list, epochs)

#繪製testing accuracy plot
plot_test_acc(test_acc_list, epochs)

#Training基本設定
parser = argparse.ArgumentParser()
parser.add_argument("-num_epochs", type=int, default=500)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-lr", type=float, default=0.01)
args = parser.parse_args()

#Train model
model = EEGNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.01)
