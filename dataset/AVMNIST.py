import torch
import os
import numpy as np
from torch.utils.data import Dataset

class AVMNIST(Dataset):
    def __init__(self, data_root='/home/rakib/Multimodal-Datasets/AV-MNIST/avmnist', mode='train'):
        super(AVMNIST, self).__init__()
        image_data_path = os.path.join(data_root, 'image')
        audio_data_path = os.path.join(data_root, 'audio')
        
        if mode == 'train':
            self.image = np.load(os.path.join(image_data_path, 'train_data.npy'))
            self.audio = np.load(os.path.join(audio_data_path, 'train_data.npy'))
            self.label = np.load(os.path.join(data_root, 'train_labels.npy'))
            
        elif mode == 'test':
            self.image = np.load(os.path.join(image_data_path, 'test_data.npy'))
            self.audio = np.load(os.path.join(audio_data_path, 'test_data.npy'))
            self.label = np.load(os.path.join(data_root, 'test_labels.npy'))

        self.length = len(self.image)
        
    def __getitem__(self, idx):
        # Get image and audio for the index
        image = self.image[idx]
        audio = self.audio[idx]
        label = self.label[idx]
        
        # Normalize image and audio
        image = image / 255.0
        audio = audio / 255.0
        
        # Reshape image and audio
        image = image.reshape(28, 28)  # Reshape to 28x28 for MNIST
        image = np.expand_dims(image, 0)  # Add channel dimension: (1, 28, 28)
        audio = np.expand_dims(audio, 0)  # Add channel dimension: (1, 28, 28)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        audio = torch.from_numpy(audio).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # Return the same format as AVDataset: (spectrogram, image_n, label, idx)
        return audio, image, label
    
    def __len__(self):
        return self.length