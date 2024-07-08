import os
import torch
import torch.nn as nn
import numpy as np
from our_model import DADFSM
import torchvision.transforms as transforms
from config import get_train_config
from checkpoint import load_checkpoint
from data_loaders import *
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter
from janghoondatautils import DataLoader,DADFSM_trainloader,DADFSM_testloader
from sklearn.metrics import *
import torch.utils.data as data
import pdb
import glob
from torch.autograd import Variable
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
from PIL import Image

import argparse

def roc_curve_plot(y_test , pred_proba_c1):
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    plt.plot(fprs , tprs, label='ROC')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    J=tprs-fprs
    index=np.argmax(J)
    best_threshold=thresholds[index]
    plt.scatter(fprs[index],tprs[index],s=100,c='r',label='Best Threshold = %.4f'%best_threshold)
  
    start, end = plt.xlim()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()

print("torch seed:",torch.initial_seed())
print("torch cuda seed:",torch.cuda.initial_seed())

scenename='Bike Roundabout'

torch.cuda.set_device(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loss_func_mse = nn.MSELoss(reduction='mean')
#loss_func_gdl=GradientDifferenceLoss()

def train_epoch(epoch, model, data_loader, batch_size, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()
    average_loss = []
    # training loop
    os.makedirs(os.path.join(save_path,"train_results_epoch%d"%epoch),exist_ok=True)
    os.makedirs(os.path.join(save_path,"train_results_epoch%d_gt"%epoch),exist_ok=True)
    for batch_idx, (batch_data) in enumerate(data_loader):
        batch_data_256, batch_data,seg_data_256,seg_data,batch_label = batch_data['256'].to(device), batch_data['standard'].to(device),batch_data['seg_256'].to(device),batch_data['seg'].to(device),batch_data['label'].numpy()

        optimizer.zero_grad()
        batch_pred = model(seg_data[:,:4],batch_data[:,:4])
        for i in range(batch_pred.shape[0]):
            predimg = batch_pred[i] 
            predimg=predimg.permute(1,2,0).detach().cpu().numpy()
            predimg = (predimg + 1) / 2
            plt.imsave(os.path.join(save_path, "train_results_epoch%d"%epoch,str(batch_idx)+ "batch%d "%i + '.jpg'), predimg)
        for j in range(seg_data_256[:,4].shape[0]):
            gtimg = seg_data_256[:,4][j]  
            gtimg=gtimg.permute(1,2,0).detach().cpu().numpy()
            gtimg = (gtimg + 1) / 2
            plt.imsave(os.path.join(save_path, "train_results_epoch%d_gt"%epoch,str(batch_idx)+ "batch%d "%j + '.jpg'), gtimg)    
        
        difference=(seg_data_256[:,4].float()-batch_pred)**2
        loss_matrix = torch.ones_like(difference, requires_grad=True).to(device)
        loss_matrix = loss_matrix * difference


        for k in range(batch_label.shape[0]):
            det_matrix=batch_label[k,4,0,:,:]
            length=int(det_matrix[0][0])
            for leng in range(length):
                x1,y1,x2,y2=det_matrix[leng+1][:4]
                loss_matrix[k,:,int(y1):int(y2),int(x1):int(x2)]+=loss_matrix[k,:,int(y1):int(y2),int(x1):int(x2)] * 1.5
        loss=loss_matrix.mean()

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

def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    #new_label = np.load('/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/UIT-ADrone-Segmented/test/test_frame_mask/DJI_0073.npy')

    with torch.no_grad():
        for batch_idx, (batch_data) in enumerate(data_loader):
            batch_data_256, batch_data = batch_data['256'].to(device), batch_data['standard'].to(device)
            # batch_target = batch_target.to(device)
            batch_pred = model(batch_data[:,:4])
            loss = loss_func_mse(batch_data_256[:,4].float(), batch_pred)
            losses.append(loss.item())

    loss = np.mean(losses)
    frame_auc = roc_auc_score(y_true=new_label[:len(losses)], y_score=losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', frame_auc)
    # metrics.update('acc5', acc5)
    print("Test Epoch: {:03d}), AUC@1: {:.2f}".format(epoch, frame_auc))
    return metrics.result()

from tqdm import tqdm
save_path = f'Drone_anomaly_results/{scenename}_newloss/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

def test_all_scenes(model, test_path,seg_path, config,current,device=None,):
    # os.makedirs('./' + save_path + 'checkpoints', exist_ok=True)
    path_ckpt = './' + save_path + 'checkpoints/current.pth'
    #path_ckpt='/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/UIT-Adrone-anomaly-detection-main/Drone_anomaly_results/Vehicle Roundabout_new/checkpoints/best.pth'
    checkpoint = torch.load(path_ckpt)
    print('Path checkpoint:', path_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    seg_scenes = glob.glob(os.path.join(seg_path, 'frames/*'))
    path_scenes = glob.glob(os.path.join(test_path, 'frames/*'))
    path_labels = f'/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/{scenename}/label'
    list_np_labels = []
    epoch=current
    losses = []
    for idx_video, path_scene in enumerate(path_scenes):
        print('------------------------------------------------------------------------------------------------------')
        print('Number of video:', idx_video+1)
        losses_curr_video = [] 
        test_dataset = DADFSM_testloader(path_scene,seg_scenes[idx_video], transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)
        os.makedirs(os.path.join(save_path, path_scene.split('/')[-1]+"_epoch"+str(epoch)), exist_ok=True)
        os.makedirs(os.path.join(save_path, path_scene.split('/')[-1]+"_epoch"+str(epoch)+"_gt"), exist_ok=True)

        test_size = len(test_dataset)
        print('Test size scene ' + path_scene.split('/')[-1] + ': %d' % test_size)
        test_batch = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
        
        np_label = np.load(os.path.join(path_labels, path_scene.split('/')[-1] + '.npy'), allow_pickle=True)
        #print(np_label)
        with torch.no_grad():
            with tqdm(desc='Evaluating ' + path_scene.split('/')[-1], unit='it', total=len(test_batch)) as pbar:
                for batch_idx, (batch_data) in enumerate(test_batch):
                    batch_data_256, batch_data,batch_seg_256,batch_seg = batch_data['256'].to(device), batch_data['standard'].to(device),batch_data['seg_256'].to(device), batch_data['seg'].to(device)
                    batch_pred = model(batch_seg[:,:4],batch_data[:, :4])
                    #print(batch_pred)labels
                    img=batch_pred.squeeze(0)
                    npimg=img.permute(1,2,0).cpu().numpy()
                    npimg = (npimg + 1) / 2

                    gtimg=batch_seg_256[:,4].float().squeeze(0)
                    gtnpimg=gtimg.permute(1,2,0).cpu().numpy()
                    gtnpimg = (gtnpimg + 1) / 2

                    plt.imsave(os.path.join(save_path, path_scene.split('/')[-1]+"_epoch"+str(epoch),str(batch_idx) + '.jpg'), npimg)
                    plt.imsave(os.path.join(save_path, path_scene.split('/')[-1]+"_epoch"+str(epoch)+'_gt',str(batch_idx) + '.jpg'), gtnpimg)
                    #loss_gdl=loss_func_gdl(batch_pred,batch_data_256[:,4].float())
                    loss_mse= loss_func_mse(batch_pred,batch_seg_256[:,4].float()) 
                    #loss=loss_mse+loss_gdl
                    loss=loss_mse
                    losses.append(loss.item())

                    losses_curr_video.append(loss.item()) # For visualization
                    pbar.update()

            list_np_labels.append(np_label[len(np_label) - len(losses_curr_video):])
        
        np.save(os.path.join(save_path, path_scene.split('/')[-1]+"_epoch%d"%epoch + '.npy'), np.array(losses_curr_video))

    list_np_labels = np.concatenate(list_np_labels)
    loss_all = np.mean(losses)
    #roc_curve_plot(list_np_labels, losses)
    print("threshold:", np.mean(losses) + np.std(losses))
    frame_auc = roc_auc_score(y_true=list_np_labels, y_score=losses)
    print("Evaluation results:, AUC@1: {:.4f} - Mean loss: {:.4f}".format(frame_auc, loss_all))
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
        self.img_size=(224,224)
        self.in_channels= 3
        self.method= "resnet50"
        #self.method='convnext'
        self.num_frames=4
        self.trans_linear_in_dim=2048#512
        self.pretrained = True


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
        train_folder = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/{scenename}/train/frames/" #f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/train/frames/"
        seg_folder = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/train/frames/"
        label_folder=f"/home/elnaggar/elnaggarprojects/drone video anomaly/drone-anomaly-segmentation/janghoon/runs/detect/{scenename}"

        print(os.listdir(train_folder))
        train_dataset = DADFSM_trainloader(train_folder,seg_folder,label_folder,transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=config.image_size, resize_width=config.image_size)

        train_size = len(train_dataset)
        print('train size: %d' % train_size)
        train_batch = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                    shuffle=False, num_workers=4, drop_last=True)

        test_folder = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/{scenename}/test/frames/" #f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/test/frames/"
        seg_test_folder = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/test/frames/"

        test_dataset = DADFSM_testloader(test_folder,seg_test_folder, transforms.Compose([
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
        batch_size = config.batch_size
        for epoch in range(1, epochs + 1):
            log['epoch'] = epoch

            # train the model
            model.train()
            result = train_epoch(epoch, model, train_batch, batch_size, optimizer, lr_scheduler, train_metrics, device)
            log.update(result)
            #f.write("Epoch: %d\n"%epoch)
            # validate the model
            if epoch >= 1:
                save_model(save_path + '/checkpoints/' , epoch, model, optimizer, lr_scheduler, device_ids, False)

                model.eval()

                test_folder = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/{scenename}/test" #f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/test"
                seg_folder  = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/test"
                frame_auc=test_all_scenes(model, test_folder, seg_folder, config, current=epoch,device='cuda')
                if frame_auc > best_acc:
                    best_acc = frame_auc
                    save_model(save_path + '/checkpoints/' , epoch, model, optimizer, lr_scheduler, device_ids, True)
                # model.eval()
                # result = valid_epoch(epoch, model, test_batch, criterion, valid_metrics, device)
                # log.update(**{'val_' + k: v for k, v in result.items()})
            else:
                save_model(save_path + '/checkpoints/' , epoch, model, optimizer, lr_scheduler, device_ids, True)


            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
    
    else:
        print('Testing ...')
        test_folder = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_order/{scenename}/test/" #f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/test/"
        seg_folder  = f"/media/elnaggar/4390958e-5b2c-4102-a19e-f2765e318006/Drone_anomaly_seg_order/{scenename}/test/"
        test_all_scenes(model, seg_folder,test_folder, config, device='cuda')

if __name__ == '__main__':
    main()
