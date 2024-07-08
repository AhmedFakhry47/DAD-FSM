from utils import setup_device, accuracy, MetricTracker, TensorboardWriter
from data_utils import DataLoader,ahjang_dataloader
import torchvision.transforms as transforms
from checkpoint import load_checkpoint
from torch.autograd import Variable
from config import get_train_config
from our_model import DADFSM
import torch.utils.data as data
from sklearn.metrics import *
from data_loaders import *
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os
import pdb
import glob


save_path = 'Drone_anomaly_results/Bike Roundabout/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

torch.cuda.set_device(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loss_func_mse = nn.MSELoss(reduction='mean')

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()
    average_loss = []
    # training loop
    for batch_idx, (batch_data) in enumerate(data_loader):
        batch_data_256, batch_data,seg_data,seg_data_256 = batch_data['256'].to(device), batch_data['standard'].to(device),batch_data['seg'].to(device), batch_data['seg_256'].to(device)
        optimizer.zero_grad()
        batch_pred = model(batch_data[:,:4],seg_data[:,:4])
        loss = loss_func_mse(seg_data_256[:,4].float(), batch_pred)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        average_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Reconstruction Loss: {:.4f}"
                    .format(epoch, batch_idx, len(data_loader), np.mean(average_loss)))
        
    return metrics.result()



def test_all_scenes(model, seg_path,test_path, config, device=None):
    path_ckpt = './' + save_path + 'checkpoints/best.pth'
    checkpoint = torch.load(path_ckpt)
    print('Path checkpoint:', path_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    seg_scenes = glob.glob(os.path.join(seg_path, 'frames/*'))
    path_scenes = glob.glob(os.path.join(test_path, 'frames/*'))

    path_labels = '/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/Bike Roundabout/label'
    list_np_labels = []

    losses = []
    for idx_video, path_scene in enumerate(path_scenes):
        print('------------------------------------------------------------------------------------------------------')
        print('Number of video:', idx_video+1)
        losses_curr_video = []
        test_dataset = ahjang_dataloader(seg_scenes[idx_video],path_scene, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)

        test_size = len(test_dataset)
        print('Test size scene ' + path_scene.split('/')[-1] + ': %d' % test_size)
        test_batch = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
        
        np_label = np.load(os.path.join(path_labels, path_scene.split('/')[-1] + '.npy'), allow_pickle=True)
        #print(np_label)
        with torch.no_grad():
            with tqdm(desc='Evaluating ' + path_scene.split('/')[-1], unit='it', total=len(test_batch)) as pbar:
                for batch_idx, (batch_data) in enumerate(test_batch):
                    batch_data_256, batch_data,batch_seg,seg_data_256 = batch_data['256'].to(device), batch_data['standard'].to(device), batch_data['seg'].to(device), batch_data['seg_256'].to(device)
                    batch_pred = model(batch_data[:, :4],batch_seg[:,:4])
                    loss = loss_func_mse(seg_data_256[:, 4].float(), batch_pred)
                    losses.append(loss.item())
                    losses_curr_video.append(loss.item()) # For visualization
                    pbar.update()

            list_np_labels.append(np_label[len(np_label) - len(losses_curr_video):])

        np.save(os.path.join(save_path, path_scene.split('/')[-1] + '.npy'), np.array(losses_curr_video))

    list_np_labels = np.concatenate(list_np_labels)
    loss_all = np.mean(losses)
    print("threshold:", np.mean(losses) + np.std(losses))
    frame_auc = roc_afuc_score(y_true=list_np_labels, y_score=losses)
    print("Evaluation results:, AUC@1: {:.2f} - Mean loss: {:.2f}".format(frame_auc, loss_all))
    return frame_auc

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str('./' + save_path + 'checkpoints/' + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str('./' + save_path + 'checkpoints/' + 'best.pth')
        torch.save(state, filename)


class trainingargs(object):
    def __init__(self):
        self.img_size= (224,224)
        self.in_channels= 3
        self.method= "resnet50"
        self.num_frames=4
        self.trans_linear_in_dim= 2048


def main():
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    args = trainingargs()
    model = DADFSM(args)

    # send model to device
    model = model.to(device)
    os.makedirs('./' + save_path + '/checkpoints', exist_ok=True)
    if bool(config.train):
        # Loading dataset
        train_folder = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/Bike Roundabout/train/frames/" 
        seg_folder = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_Anomaly_seg_order/Bike Roundabout/train/frames/"

        print(os.listdir(train_folder))
        train_dataset = ahjang_dataloader(seg_folder,train_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)

        train_size = len(train_dataset)
        print('train size: %d' % train_size)
        train_batch = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                    shuffle=True, num_workers=4, drop_last=True)

        test_folder = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/Bike Roundabout/test/frames/" 
        seg_test_folder = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_Anomaly_seg_order/Bike Roundabout/test/frames/"

        test_dataset = ahjang_dataloader(seg_test_folder,test_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)

        test_size = len(test_dataset)
        print('test size: %d' % test_size)
        test_batch = data.DataLoader(test_dataset, batch_size=1,
                                    shuffle=False, num_workers=4, drop_last=True)

        print('dataload!')

        # training criterion
        print("create criterion and optimizer")
        criterion = nn.CrossEntropyLoss()

        # create optimizers and learning rate scheduler
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.wd)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.lr,
            pct_start=config.warmup_steps / config.train_steps,
            total_steps=config.train_steps)
        

        print("start training")
        best_acc = 0.0
        log = {}
        log['val_acc1'] = 0
        # epochs = config.train_steps // len(train_dataloader)
        epochs = config.epochs
        for epoch in range(1, epochs + 1):
            log['epoch'] = epoch

            # train the model
            model.train()
            result = train_epoch(epoch, model, train_batch, criterion, optimizer, lr_scheduler, train_metrics, device)
            log.update(result)

            # validate the model
            if epoch >= 1:
                save_model(save_path + '/checkpoints/' , epoch, model, optimizer, lr_scheduler, device_ids, True )

                model.eval()

                test_folder = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/Bike Roundabout/test" 
                seg_folder  = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_Anomaly_seg_order/Bike Roundabout/test"
                test_all_scenes(model, seg_folder,test_folder, config, device='cuda')
            else:
                save_model(save_path + '/checkpoints/' , epoch, model, optimizer, lr_scheduler, device_ids, True)

            # best acc
            best = False
            if log['val_acc1'] > best_acc:
                best_acc = log['val_acc1']
                best = True

            # save model
            save_model(save_path + '/checkpoints/' , epoch, model, optimizer, lr_scheduler, device_ids, best)

            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
    
    else:
        print('Testing ...')
        test_folder = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/Bike Roundabout/test/" 
        seg_folder  = "/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_Anomaly_seg_order/Bike Roundabout/test/"
        test_all_scenes(model, seg_folder,test_folder, config, device='cuda')

if __name__ == '__main__':
    main()
