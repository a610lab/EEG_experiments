import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from scipy.io import loadmat
from ssl_dataset import MyDataset
from model_file.EEGNet import eegnet
from model_file.ShallowNet import shallow_net
from ssl_method import rearrange
from torch.utils.data.dataloader import DataLoader
from process_data import process_BCICIV_data_set_1

fix_seed = 1029

def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_seeds():
    
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    init_seeds()
    torch.set_num_threads(1)

    train_batch_size = 8
    train_epochs = 200
    model_name = 'ShallowNet'
    subject = 'a'
    learning_rate = 0.0001

    mat_file_path = './data_set/BCICIV/data_set_1/100Hz/BCICIV_calib_ds1{0}.mat'.format(subject)

    mat_file = loadmat(mat_file_path)

    X, _ = process_BCICIV_data_set_1(mat_file, low_frequency_limit=8, high_frequency_limit=30, random_seed=fix_seed)

    slice_nums = 4
    times = 1

    X = torch.from_numpy(X)
    X, y, index = rearrange(X, slice_nums, times)

    X = torch.unsqueeze(X, dim=1)

    train_set = MyDataset(X, y, index)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=0)

    learning_rate_str = '1e-4'

    save_path = './save_ssl_model/{0}_{1}_{2}e_{3}b_{4}lr_ssl.pth'.format(subject, model_name, train_epochs,
                                                                          train_batch_size, learning_rate_str)

    net = shallow_net()
    device = torch.device('cuda:0')
    net.to(device)
	
	net.apply(init_weight)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate)

    net.train()
    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(train_epochs):
        running_acc = 0.0
        mean_loss = torch.zeros(1).to(device)
        train_bar = tqdm(train_loader)

        for step, data in enumerate(train_bar):
            train_data, train_labels = data
            train_data = train_data.float().to(device)
            train_labels = train_labels.type(torch.LongTensor).to(device)

            outputs = net(train_data)
            loss = loss_function(outputs, train_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predict_labels = torch.max(outputs, dim=1)[1]

            running_acc = running_acc + torch.eq(predict_labels, train_labels).sum().item()
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

            train_bar.desc = "train epoch[{0}/{1}] loss:{2:.3f} acc:{3:.3f}".format(epoch + 1,
                                                                                    train_epochs,
                                                                                    mean_loss.item(),
                                                                                    running_acc / len(index))

        mean_running_loss = mean_loss.item()
        if mean_running_loss < best_loss:
            best_loss = mean_running_loss
            torch.save(net.state_dict(), save_path)

    print('minimum train loss:{0:.3f} maximum train accuracy:{1:.3f}'.format(best_loss, best_acc))