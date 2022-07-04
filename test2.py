import torch
import numpy as np
import matplotlib.pyplot as plt
import mne

from scipy.io import loadmat
from my_dataset import MyDataset
from model_file.EEGNet import eegnet
from model_file.Resnet import resnet34_rearrange
from model_file.MobileNet import mobilenetV2_rearrange
from model_file.EfficientNet import efficientnet_b0_ra
from model_file.Deep4 import deep4
from model_file.ShallowNet import shallow_net
from braindecode.models.deep4 import Deep4Net
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from process_data import process_BCICIV_data_set_2b
from sklearn.metrics import roc_curve, auc, cohen_kappa_score


def init_seeds(fix_seed):
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_result(predict_labels: np.ndarray, true_labels: np.ndarray, predict_probability: np.ndarray):
    result = {}

    acc = np.sum(predict_labels == true_labels)

    true_positive_labels = np.dot(predict_labels, true_labels.T)
    true_negative_labels = acc - true_positive_labels

    positive_labels = np.sum(true_labels)
    negative_labels = len(true_labels) - positive_labels
	
    result['acc'] = acc / true_labels.shape[0]
    result['pos_acc'] = true_positive_labels / positive_labels
    result['neg_acc'] = true_negative_labels / negative_labels

    kappa_value = cohen_kappa_score(true_labels, predict_labels)
    result['kappa'] = kappa_value

    fpr, tpr, thresholds = roc_curve(true_labels, predict_probability[:, 1])
    x_fpr = np.linspace(0, 1, 100)
    y_tpr = np.interp(x_fpr, fpr, tpr)
    y_tpr[0] = 0
    y_tpr[-1] = 1
    result['tpr'] = y_tpr

    auc_value = auc(fpr, tpr)
    result['auc'] = auc_value

    return result

def test(seed, flag):
    train_batch_size = 50
    train_epochs = 100
    model_name = 'ShallowNet'
    use_ssl_flag = flag
    subject = '9'
    few_percent = 0.5
    learning_rate = 0.001
    print(subject, few_percent)


    gdf_file_path_1 = './data_set/BCICIV/data_set_2b/B0{0}01T.gdf'.format(subject)
    gdf_file_1 = mne.io.read_raw_gdf(gdf_file_path_1, preload=True)
    gdf_file_path_2 = './data_set/BCICIV/data_set_2b/B0{0}02T.gdf'.format(subject)
    gdf_file_2 = mne.io.read_raw_gdf(gdf_file_path_2, preload=True)

    X_1, Y_1 = process_BCICIV_data_set_2b(gdf_file_1, random_seed=seed)
    X_2, Y_2 = process_BCICIV_data_set_2b(gdf_file_2, random_seed=seed)
    X = np.append(X_1, X_2, axis=0)
    Y = np.append(Y_1, Y_2, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed, stratify=Y)

    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)

    X_test = torch.unsqueeze(X_test, dim=1)

    test_set = MyDataset(X_test, Y_test)
    test_loader = DataLoader(test_set, batch_size=train_batch_size, shuffle=True, num_workers=0)

    few_percent_str = '100' if few_percent == -1 else (str(int(few_percent * 100)))
    learning_rate_str = '1e-3'

    load_path = './save_model/{0}_{1}_{2}e_{3}p_{4}b_{5}lr_{6}s'.format(subject, model_name, train_epochs,
                                                                        few_percent_str, train_batch_size,
                                                                        learning_rate_str, seed)
    if use_ssl_flag:
        load_path = load_path + '_ssl.pth'
    else:
        load_path = load_path + '.pth'

    net = shallow_net()
    device = torch.device('cuda:0')
    net.to(device)

    net.load_state_dict(torch.load(load_path))

    predict_labels = []
    true_labels = []
    all_outputs = None

    net.eval()

    with torch.no_grad():
        for step, data in enumerate(test_loader):
            test_data, test_labels = data
            test_data = test_data.float().to(device)
            test_labels = test_labels.cpu().numpy().tolist()

            outputs = net(test_data)

            if all_outputs is None:
                all_outputs = outputs
            else:
                all_outputs = torch.cat((all_outputs, outputs), dim=0)

            part_labels = torch.max(outputs, dim=1)[1]
            predict_labels = predict_labels + part_labels.cpu().numpy().tolist()
            true_labels = true_labels + test_labels

    predict_probability = torch.softmax(all_outputs, dim=1)
    predict_probability = predict_probability.cpu().numpy()

    result = calculate_result(np.array(predict_labels), np.array(true_labels), predict_probability)

    print('\taccuracy:{0:.3f}, positives accuracy:{1:.3f}, negatives accuracy:{2:.3f}, kappa:{3:.3f}, auc:{4:.3f} \n'
          .format(result['acc'], result['pos_acc'], result['neg_acc'], result['kappa'], result['auc']))

    plt.plot([0, 1], [0, 1], lw=1, color='r')
    plt.plot(np.linspace(0, 1, 100), result['tpr'], label=r'ROC Curve', lw=2, linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
    plt.close()
    return result['acc'], result['auc']
