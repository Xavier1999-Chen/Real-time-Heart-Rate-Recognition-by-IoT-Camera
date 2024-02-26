"""Trainer for Physformer.

Based on open-source code from the original PhysFormer authors below:
https://github.com/ZitongYu/PhysFormer/blob/main/train_Physformer_160_VIPL.py

We also thank the PhysBench authors for their open-source code based on the code
of the original authors. Their code below provided a better reference for tuning loss
parameters of interest and utilizing RSME as a validation loss:
https://github.com/KegangWangCCNU/PhysBench/blob/main/benchmark_addition/PhysFormer_pure.ipynb

"""

import os
import numpy as np
import math
import torch
import torch.optim as optim
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from scipy.signal import welch

class PhysFormerTrainer(BaseTrainer):

    def __init__(self):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device('cpu:0')
        self.dropout_rate = 0.2
        self.patch_size = 4
        self.dim = 96
        self.ff_dim = 144
        self.num_heads = 4
        self.num_layers = 12
        self.theta = 0.7
        # self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        # self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = 1
        self.chunk_len = 160
        # self.frame_rate = config.TRAIN.DATA.FS
        # self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0
        self.release_model_path="./final_model_release/PURE_PhysFormer_DiffNormalized.pth"

        
        
        self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
            image_size=(self.chunk_len,128,128), 
            patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
            dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(1)))
    

# input ([4, 3, 160, 128, 128])
    def test(self, input):
        #print("===Testing===")

        #if self.config.TOOLBOX_MODE == "only_test":
        if not os.path.exists(self.release_model_path):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        self.model.load_state_dict(torch.load(self.release_model_path, map_location=torch.device('cpu')))
        #print("Testing uses pretrained model!")
       

        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            data= input.to(self.device)
            gra_sharp = 2.0
            #print('input_shape',data.shape)
            pred_ppg_test, _, _, _ = self.model(data, gra_sharp)
            #print('pred_ppg',pred_ppg_test.shape)
        return pred_ppg_test
        # print('')
        # calculate_metrics(predictions, labels, self.config)
        # if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
        #     self.save_test_outputs(predictions, labels, self.config)

    # HR calculation based on ground truth label
    # def get_hr(self, y, sr=30, min=30, max=180):
    #     p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
    #     return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
