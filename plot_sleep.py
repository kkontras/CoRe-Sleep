import sys
sys.exc_info()
from utils.config import process_config
from datasets.sleepset import *
from graphs.models.attention_models.windowFeature_base import *
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_predictions_time_series(model, views, inits):
    """
    This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
    :param views: List of tensors, data views/modalities
    :param inits: Tensor indicating with value one, when there incontinuities.
    :return: predictions of the model on the batch
    """
    inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
    if len(inits_sum) > 0:
        batch = views[0].shape[0]
        outer = views[0].shape[1]
        batch_idx_checked = torch.ones(batch, dtype=torch.bool)
        pred = torch.zeros(batch * outer, 5).cuda()
        for idx in inits_sum:
            if inits[idx].sum() > 1:
                ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                if (ones_idx[0] + 1 == ones_idx[1]):  # and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                    if ones_idx[0] == 0:
                        pred_split_0 = model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                    else:
                        pred_split_0 = model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                    if ones_idx[1] == len(inits[idx]):
                        pred_split_1 = model([view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                    else:
                        pred_split_1 = model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                    pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                    batch_idx_checked[idx] = False
                else:
                    pred[idx * outer:(idx + 1) * outer] = model([view[idx].unsqueeze(dim=0) for view in views])
        pred[batch_idx_checked.repeat_interleave(outer)] = model([view[batch_idx_checked] for view in views])
    else:
        pred = model(views)
    return pred
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)
def test(model,data_loader):
    model.eval()
    with torch.no_grad():
        tts, preds, inits = [], [], []
        pbar = tqdm(enumerate(data_loader.test_loader), desc="Test")
        for batch_idx, (data, target, init, _) in pbar:
            views = [data[i].float().to(device) for i in range(len(data))]
            label = target.to(device).flatten()
            pred = get_predictions_time_series(model, views, init)
            tts.append(label)
            preds.append(pred)
            inits.append(init.flatten())
            pbar.set_description("Test batch {0:d}/{1:d}".format(batch_idx, len(data_loader.test_loader)))
            pbar.refresh()
        tts = torch.cat(tts).cpu().numpy()
        preds = torch.cat(preds).cpu().numpy()

    multiclass = False
    if preds.shape[1] > 2:
        multiclass = True
    preds = preds.argmax(axis=1)
    test_acc = np.equal(tts, preds).sum() / len(tts)
    test_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average="macro")
    test_perclass_f1 = f1_score(preds, tts) if not multiclass else f1_score(preds, tts, average=None)
    test_k = cohen_kappa_score(tts, preds)
    test_auc = roc_auc_score(tts, preds) if not multiclass else 0
    test_conf = confusion_matrix(tts, preds)
    tp, fp, tn, fn = perf_measure(tts, preds)
    test_spec = tn / (tn + fp)
    test_sens = tp / (tp + fn)
    print("Test accuracy: {0:.2f}% f1 :{1:.4f}, k :{2:.4f}, sens:{3:.4f}, spec:{4:.4f}, f1_per_class :{5:40}".format(
            test_acc * 100,
            test_f1,
            test_k, test_spec, test_sens,
            "{}".format(list(test_perclass_f1))))
    return test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens
def plot_hypnogram(data_loader, patient_num, model, device):
    data_loader.test_loader.dataset.choose_specific_patient(patient_num)
    data, target, inits, idxs = next(iter(data_loader.test_loader))
    views = [data[i].float().to(device) for i in range(len(data))]
    target = target.to(device).flatten()
    pred = get_predictions_time_series(model, views, inits)
    pred = pred.argmax(axis=1)

    target = target + 0.05
    target = target[:60 * 2 * 3]
    pred = pred[:60 * 2 * 3]
    hours = len(target)
    plt.plot(pred.detach().cpu().numpy())
    plt.plot(target.detach().cpu().numpy())
    plt.yticks([0, 1, 2, 3, 4], labels=["Wake", "N1", "N2", "N3", "REM"])
    plt.xticks([i * 120 for i in range((hours // 120) + 1)],
               labels=["{}_hours".format(i) for i in range((hours // 120) + 1)])
    plt.legend(["Prediction", "Label"])
    plt.show()

def sleep_plot_losses(config, logs):
    train_loss = np.array([logs["train_logs"][i]["train_loss"] for i in logs["train_logs"]])
    val_loss = np.array([logs["val_logs"][i]["val_loss"]for i in logs["val_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    plt.figure()
    plt.plot(steps, train_loss, label="Train")
    plt.plot(steps, val_loss, label="Valid")

    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]-1
    best_loss = logs["best_logs"]["val_loss"]

    plt.plot((best_step, best_step), (0, best_loss), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, best_step), (best_loss, best_loss), linestyle="--", color="y")

    if config.rec_test:
        test_loss = np.array([logs["test_logs"][i]["test_loss"] for i in logs["test_logs"]])
        best_test_step = np.argmin(test_loss)
        best_test_loss = test_loss[best_test_step]
        plt.plot(steps, test_loss, label="Test")
        plt.plot((best_test_step, best_test_step), (0, best_test_loss), linestyle="--", color="r", label="Chosen Point")
        plt.plot((0, best_test_step), (best_test_loss, best_test_loss), linestyle="--", color="r")

    plt.xlabel('Epochs')
    plt.ylabel('Loss Values')
    plt.title("Loss")
    plt.ylim([1,1.16])
    plt.legend()
    plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/loss.png")
    plt.show()

def sleep_plot_k(config, logs):

    train_k = np.array([logs["train_logs"][i]["train_k"] for i in logs["train_logs"]])
    val_k = np.array([logs["val_logs"][i]["val_k"]for i in logs["val_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    plt.figure()
    plt.plot(steps, train_k, label="Train")
    plt.plot(steps, val_k, label="Valid")

    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]-1
    best_k = logs["best_logs"]["val_k"]

    plt.plot((best_step, best_step), (0, best_k), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, best_step), (best_k, best_k), linestyle="--", color="y")

    if config.rec_test:
        test_k = np.array([logs["test_logs"][i]["test_k"] for i in logs["test_logs"]])
        best_test_step = np.argmax(test_k)
        best_test_k = test_k[best_test_step]
        plt.plot(steps, test_k, label="Test")
        plt.plot((best_test_step, best_test_step), (0, best_test_k), linestyle="--", color="r", label="Chosen Point")
        plt.plot((0, best_test_step), (best_test_k, best_test_k), linestyle="--", color="r")

    plt.xlabel('Epochs')
    plt.ylabel('Kappa')
    plt.title("Cohen's kappa")
    plt.legend()
    plt.ylim([0.2,0.8])
    plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/kappa.png")
    plt.show()

def sleep_plot_f1(config, logs):

    train_f1 = np.array([logs["train_logs"][i]["train_f1"] for i in logs["train_logs"]])
    val_f1 = np.array([logs["val_logs"][i]["val_f1"]for i in logs["val_logs"]])
    steps = np.array([i/logs["train_logs"][i]["validate_every"] for i in logs["train_logs"]])-1

    plt.figure()
    plt.plot(steps, train_f1, label="Train")
    plt.plot(steps, val_f1, label="Valid")

    best_step = logs["best_logs"]["step"]/logs["train_logs"][logs["best_logs"]["step"]]["validate_every"]-1
    best_f1 = logs["best_logs"]["val_f1"]

    plt.plot((best_step, best_step), (0, best_f1), linestyle="--", color="y", label="Chosen Point")
    plt.plot((0, best_step), (best_f1, best_f1), linestyle="--", color="y")

    if config.rec_test:
        test_f1 = np.array([logs["test_logs"][i]["test_f1"] for i in logs["test_logs"]])
        best_test_step = np.argmax(test_f1) - 1
        best_test_f1 = test_f1[best_test_step]
        plt.plot(steps, test_f1, label="Test")
        plt.plot((best_test_step, best_test_step), (0, best_test_f1), linestyle="--", color="r", label="Chosen Point")
        plt.plot((0, best_test_step), (best_test_f1, best_test_f1), linestyle="--", color="r")

    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.title("Training progress: F1 ")
    plt.ylim([0.2,0.8])
    plt.legend()
    plt.savefig("/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/f1.png")
    plt.show()

config = "./configs/shhs/fourier_transformer_eeg.json"
config = process_config(config)
config.test_batch_size = 256
config.print_statistics = False

# config.save_dir = "/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/shhs_sleepTransformer_eeg_bfc.pth.tar"


device = "cuda:{}".format(config.gpu_device[0])
dataloader = globals()[config.dataloader_class]
data_loader = dataloader(config=config)
checkpoint = torch.load(config.save_dir)
model_class = globals()[config.model_class]
model = model_class([], channel = config.channel)
model = model.to(device)
model = nn.DataParallel(model, device_ids=[torch.device(i) for i in config.gpu_device])
model.load_state_dict(checkpoint["best_model_state_dict"])

if config.save_dir == "/users/sista/kkontras/Documents/Sleep_Project/data/2021_data/shhs_sleepTransformer_eeg_bfc.pth.tar":
    train_logs = checkpoint["train_logs"]
    test_logs = checkpoint["test_logs"]
    best_logs = checkpoint["best_logs"]

    print(best_logs)

    b = {"step": best_logs[0].item(),
                              "val_loss": best_logs[1].item(),
                              "val_acc": best_logs[2].item(),
                              "val_f1": best_logs[3].item(),
                              "val_k": best_logs[4].item(),
                              "val_perclassf1": [0,0,0,0,0]}

    t = {}
    v = {}
    for i in range(1,len(train_logs)):
        if np.array(train_logs[i].cpu().numpy()).sum() ==0:
            idx = i
            print(i)
            break
        t[i] = {}
        t[i]["train_loss"] = train_logs[i][1].item()
        t[i]["train_acc"] = train_logs[i][2].item()
        t[i]["train_f1"] = train_logs[i][4].item()
        t[i]["train_k"] = train_logs[i][6].item()
        t[i]["validate_every"] = 1
        v[i] = {}
        v[i]["val_loss"] = train_logs[i][0].item()
        v[i]["val_acc"] = train_logs[i][3].item()
        v[i]["val_f1"] = train_logs[i][5].item()
        v[i]["val_k"] = train_logs[i][7].item()
    logs = {}
    logs = {"current_epoch": checkpoint["epoch"], "current_step": idx, "steps_no_improve": 18, "train_logs": t, "val_logs": v,
                 "test_logs": {}, "best_logs": b, "seed": 28 }
else:
    logs = checkpoint['logs']


# logs = checkpoint["logs"]

# test_acc, test_f1, test_k, test_auc, test_conf, test_perclass_f1, test_spec, test_sens = test(model, data_loader)
# plot_hypnogram(data_loader, 20, model, device)
sleep_plot_losses(config, logs)
sleep_plot_k(config, logs)
sleep_plot_f1(config, logs)
