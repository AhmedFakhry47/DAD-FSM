from collections import OrderedDict
import torch.utils.data as data
import numpy as np
import torch
import glob
import json
import cv2
import pdb
import os


rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width,background_sub=False):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    h, w, _ = image_decoded.shape
    
    
    if background_sub:
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        gray = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2GRAY)
        image_decoded = fgbg.apply(gray)
        image_decoded = cv2.cvtColor(image_decoded,cv2.COLOR_GRAY2RGB)

    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized, h, w

class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.video_frames = []
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.index_samples = []
        self.setup()

    def setup(self):
        print(self.dir)
        videos = glob.glob(os.path.join(self.dir, '*'))
        videos.sort()
        if os.path.isdir(videos[0]):
            all_video_frames = []
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.jpg')) #change to jpg later
                vide_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) #.split('_')[1]) # 0->-1 later .split('_')[0]
                if len(all_video_frames) == 0:
                    all_video_frames = vide_frames
                else:
                    all_video_frames += vide_frames
        else:
            videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) #.split('_')[1] # 0 -> -1 later .split('_')[0]
            all_video_frames = videos
        
        self.video_frames = all_video_frames
        self.index_samples = list(range(len(all_video_frames)-self._time_step))

    def __getitem__(self, index):
        frame_index = self.index_samples[index]
        batch_frames_512 = np.zeros((self._time_step+self._num_pred, 3, 256, 256))
        batch_frames = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))

        for i in range(self._time_step + self._num_pred):
            image_512, h, w = np_load_frame(self.video_frames[frame_index + i], 256, 256)
            image, h, w = np_load_frame(self.video_frames[frame_index + i], self._resize_height,
                                  self._resize_width)

            if self.transform is not None:
                batch_frames_512[i] = self.transform(image_512)
                batch_frames[i] = self.transform(image)
            
        return {
            '256': batch_frames_512,
            'standard': batch_frames
        }

    def __len__(self):
        return len(self.index_samples)

class DADFSM_loader(DataLoader):
    def __init__(self,video_folder,segmented_video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.seg_dir = segmented_video_folder
        self.segmented_frames = []
        self.segmented_index  = []
        super().__init__(video_folder,transform, resize_height, resize_width, time_step=4, num_pred=1)
        self.seg_setup()

        
    def seg_setup(self):
        videos = glob.glob(os.path.join(self.seg_dir, '*'))
        videos.sort()
        if os.path.isdir(videos[0]):
            all_video_frames = []
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.png')) #change to jpg later
                vide_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0])) # 0->-1 later .split('_')[0]
                if len(all_video_frames) == 0:
                    all_video_frames = vide_frames
                else:
                    all_video_frames += vide_frames
        else:
            videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0])) # 0 -> -1 later .split('_')[0]
            all_video_frames = videos
        
        self.segmented_frames = all_video_frames
        self.segmented_index  = list(range(len(all_video_frames)-self._time_step))

    def __getitem__(self, index):
        frame_index = self.index_samples[index]
        try:
            seg_index   = self.segmented_index[index]
        except:
            seg_index   = self.segmented_index[-1]

        batch_frames_512 = np.zeros((self._time_step+self._num_pred, 3, 256, 256))
        batch_frames = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))

        batch_seg_512 = np.zeros((self._time_step+self._num_pred, 3, 256, 256))
        batch_seg = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))

        for i in range(self._time_step + self._num_pred):
            image_512, h, w = np_load_frame(self.video_frames[frame_index + i], 256, 256)
            image, h, w = np_load_frame(self.video_frames[frame_index + i], self._resize_height,
                                  self._resize_width)
            
            seg_512, h, w = np_load_frame(self.segmented_frames[seg_index + i], 256, 256,background_sub=False) #Changes made here
            seg, h, w = np_load_frame(self.segmented_frames[seg_index + i], self._resize_height, 
                                  self._resize_width,background_sub=False)#Changes made here

            if self.transform is not None:
                batch_frames_512[i] = self.transform(image_512)
                batch_frames[i] = self.transform(image)

                batch_seg_512[i] = self.transform(seg_512)
                batch_seg[i] = self.transform(seg)

        return {
            '256': batch_frames_512,
            'standard': batch_frames,
            'seg_256': batch_seg_512,
            'seg': batch_seg
        }

    def __len__(self):
        return len(self.index_samples)

def load_label(label_path, resize_height, resize_width):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # Convert YOLO format to [x1, y1, x2, y2] format
            x_center *= resize_width
            y_center *= resize_height
            width *= resize_width
            height *= resize_height
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            labels.append([int(x1), int(y1), int(x2), int(y2)])
    matrix = np.zeros((3, 256, 256))

    matrix[0, 0, 0] = len(labels)

    # bounding box 좌표를 매트릭스에 저장
    for i, (x1, y1, x2, y2) in enumerate(labels):
        if i < 256:  # 좌표를 저장할 수 있는 최대 개수는 255개
            matrix[0, i + 1, :4] = [x1, y1, x2, y2]
    
    return matrix

class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.video_frames = []
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.index_samples = []
        self.setup()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        videos.sort()
        if os.path.isdir(videos[0]):
            all_video_frames = []
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.jpg')) #change to jpg later
                vide_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) #.split('_')[1] # 0->-1 later .split('_')[0]
                if len(all_video_frames) == 0:
                    all_video_frames = vide_frames
                else:
                    all_video_frames += vide_frames
        else:
            videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0])) #.split('_')[1] # 0 -> -1 later .split('_')[0]
            all_video_frames = videos
        
        self.video_frames = all_video_frames
        self.index_samples = list(range(len(all_video_frames)-self._time_step))

    def __getitem__(self, index):
        frame_index = self.index_samples[index]
        batch_frames_512 = np.zeros((self._time_step+self._num_pred, 3, 256,256)) #to change later to 256
        batch_frames = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))

        for i in range(self._time_step + self._num_pred):
            image_512, h, w = np_load_frame(self.video_frames[frame_index + i], 256, 256) # to change later to 256
            image, h, w = np_load_frame(self.video_frames[frame_index + i], self._resize_height,
                                  self._resize_width)

            if self.transform is not None:
                batch_frames_512[i] = self.transform(image_512)
                batch_frames[i] = self.transform(image)
            
        return {
            '256': batch_frames_512,
            'standard': batch_frames
        }

    def __len__(self):
        return len(self.index_samples)

class DADFSM_trainloader(DataLoader):
    def __init__(self,video_folder,segmented_video_folder,label_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.seg_dir = segmented_video_folder
        self.segmented_frames = []
        self.segmented_index  = []
        self.label_dir = label_folder
        self.label_info=[]
        self.label_index=[]
        super().__init__(video_folder,transform, resize_height, resize_width, time_step=4, num_pred=1)
        self.seg_setup()

        
    def seg_setup(self):
        videos = glob.glob(os.path.join(self.seg_dir, '*'))
        videos.sort()
        labels=[os.path.join(self.label_dir,i) for i in os.listdir(self.label_dir)] #glob.glob(os.path.join(self.label_dir, '*'))        
        if os.path.isdir(videos[0]):
            all_video_frames = []
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.png')) #change to jpg later
                vide_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0])) # 0->-1 later .split('_')[0]
                if len(all_video_frames) == 0:
                    all_video_frames = vide_frames
                else:
                    all_video_frames += vide_frames
        else:
            videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0])) # 0 -> -1 later .split('_')[0]
            all_video_frames = videos

        if os.path.isdir(labels[0]):
            all_label_info = []
            for label in labels:
                label_info = [os.path.join(label,i) for i in os.listdir(label) if '.txt' in i] #glob.glob(os.path.join(label, '*.txt'))
                label_info.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1])) #Change this back from 1 -> 0
                if len(all_label_info) == 0:
                    all_label_info = label_info
                else:
                    all_label_info += label_info
        else:
            labels.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0]))
            all_label_info = labels

        self.segmented_frames = all_video_frames
        self.segmented_index  = list(range(len(all_video_frames)-self._time_step))
        self.label_info=all_label_info
        self.label_index=list(range(len(all_label_info)-self._time_step))

    def __getitem__(self, index):
        frame_index = self.index_samples[index]
        try:
            seg_index   = self.segmented_index[index]
        except:
            seg_index   = self.segmented_index[-1]
        try:
            label_index = self.label_index[index]
        except:
            label_index = self.label_index[-1]
        #print(self.label_index)
        batch_frames_512 = np.zeros((self._time_step+self._num_pred, 3, 256, 256))
        batch_frames = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))

        batch_seg_512 = np.zeros((self._time_step+self._num_pred, 3, 256, 256))
        batch_seg = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))
        #print(self.label_info)

        batch_labels = np.zeros((self._time_step + self._num_pred, 3, 256, 256))

        for i in range(self._time_step + self._num_pred):
            image_512, h, w = np_load_frame(self.video_frames[frame_index + i], 256, 256)
            image, h, w = np_load_frame(self.video_frames[frame_index + i], self._resize_height,
                                  self._resize_width)
            
            seg_512, h, w = np_load_frame(self.segmented_frames[seg_index + i], 256, 256)
            seg, h, w = np_load_frame(self.segmented_frames[seg_index + i], self._resize_height,
                                  self._resize_width)

            label=load_label(self.label_info[label_index + i],256,256)

            if self.transform is not None:
                batch_frames_512[i] = self.transform(image_512)
                batch_frames[i] = self.transform(image)

                batch_seg_512[i] = self.transform(seg_512)
                batch_seg[i] = self.transform(seg)
            
            batch_labels[i] = label
        
        #print(batch_labels.shape)
        return {
            '256': batch_frames_512,
            'standard': batch_frames,
            'seg_256': batch_seg_512,
            'seg': batch_seg,
            'label':batch_labels
        }
    def __len__(self):
        return len(self.index_samples)


class DADFSM_testloader(DataLoader):
    def __init__(self,video_folder,segmented_video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.seg_dir = segmented_video_folder
        self.segmented_frames = []
        self.segmented_index  = []
        super().__init__(video_folder,transform, resize_height, resize_width, time_step=4, num_pred=1)
        self.seg_setup()

        
    def seg_setup(self):
        videos = glob.glob(os.path.join(self.seg_dir, '*'))
        videos.sort()
        if os.path.isdir(videos[0]):
            all_video_frames = []
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.png')) #change to jpg later
                vide_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0])) # 0->-1 later .split('_')[0]
                if len(all_video_frames) == 0:
                    all_video_frames = vide_frames
                else:
                    all_video_frames += vide_frames
        else:
            videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0])) # 0 -> -1 later .split('_')[0]
            all_video_frames = videos


        
        self.segmented_frames = all_video_frames
        self.segmented_index  = list(range(len(all_video_frames)-self._time_step))

    def __getitem__(self, index):
        frame_index = self.index_samples[index]
        try:
            seg_index   = self.segmented_index[index]
        except:
            seg_index   = self.segmented_index[-1]

        batch_frames_512 = np.zeros((self._time_step+self._num_pred, 3, 256, 256))
        batch_frames = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))

        batch_seg_512 = np.zeros((self._time_step+self._num_pred, 3, 256, 256))
        batch_seg = np.zeros((self._time_step+self._num_pred, 3, self._resize_width, self._resize_height))

        for i in range(self._time_step + self._num_pred):
            image_512, h, w = np_load_frame(self.video_frames[frame_index + i], 256, 256)
            image, h, w = np_load_frame(self.video_frames[frame_index + i], self._resize_height,
                                  self._resize_width)
            
            seg_512, h, w = np_load_frame(self.segmented_frames[seg_index + i], 256, 256)
            seg, h, w = np_load_frame(self.segmented_frames[seg_index + i], self._resize_height,
                                  self._resize_width)

            if self.transform is not None:
                batch_frames_512[i] = self.transform(image_512)
                batch_frames[i] = self.transform(image)

                batch_seg_512[i] = self.transform(seg_512)
                batch_seg[i] = self.transform(seg)
  
        return {
            '256': batch_frames_512,
            'standard': batch_frames,
            'seg_256': batch_seg_512,
            'seg': batch_seg
        }

    def __len__(self):
        return len(self.index_samples)