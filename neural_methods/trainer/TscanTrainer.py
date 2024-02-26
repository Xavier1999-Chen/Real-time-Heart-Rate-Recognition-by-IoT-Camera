"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class TscanTrainer(BaseTrainer):

    def __init__(self):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device('cpu:0')
        self.frame_depth = 10
        # self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        # self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = 1
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = 180
        # self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0
        self.release_model_path="./final_model_release/PURE_TSCAN.pth"
        
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=72).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(1)))
    

    def test(self, input):
        # print("===Testing===")

        # if self.config.TOOLBOX_MODE == "only_test":
        if not os.path.exists(self.release_model_path):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        self.model.load_state_dict(torch.load(self.release_model_path, map_location=torch.device('cpu')))
        # print("Testing uses pretrained model!")
   

        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            data_test = input.to(self.device)
            N, D, C, H, W = data_test.shape
            data_test = data_test.view(N * D, C, H, W)
            # labels_test = labels_test.view(-1, 1)
            data_test = data_test[:(N * D) // self.base_len * self.base_len]
            # labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
            pred_ppg_test = self.model(data_test)
        return pred_ppg_test


    #     print('')
    #     calculate_metrics(predictions, labels, self.config)
    #     if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
    #         self.save_test_outputs(predictions, labels, self.config)

    # def save_model(self, index):
    #     if not os.path.exists(self.model_dir):
    #         os.makedirs(self.model_dir)
    #     model_path = os.path.join(
    #         self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
    #     torch.save(self.model.state_dict(), model_path)
    #     print('Saved Model Path: ', model_path)
