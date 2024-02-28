import subprocess
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import time
import json
import faiss
from pathlib import Path
from PIL import Image, ImageFilter

from torch import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


Z_DIM = 500
HEIGHT = 80
WIDTH = 96

# shellで、kodakスキャナーを動かして画像を取得
def scan_imgs():
    subprocess.run('rm imgs/*', shell=True)
    subprocess.run('scanimage --mode=color --format=png --batch="imgs/scan_%03d.png" --batch-start=1 --batch-count=100', shell=True)


# 画像の期待値を0.5, sdを0.2に変換
def image_norm(img: np.array) -> np.array:
    mu, sd = np.mean(img, axis =(0,1)), np.std(img, axis = (0,1))
    img = 0.5 + 0.2 * (img - mu)/sd
    return(img)


# ベクトル化用のオートエンコーダ・モデルの定義
class AE(nn.Module):
    def __init__(self, z_dim):
      super(AE, self).__init__()
      self.conv2d_enc1 = nn.Conv2d(3, 16, kernel_size=2,stride = 2, padding=0, bias=True)
      self.conv2d_enc2 = nn.Conv2d(16, 32, kernel_size=2,stride = 2, padding=0, bias=True)
      self.conv2d_enc3 = nn.Conv2d(32, 32, kernel_size=2,stride = 2, padding=0, bias=True)
      self.flatten_enc1 = nn.Flatten()

      self.dense_enc1 = nn.Linear(int(HEIGHT/8)*int(WIDTH/8)*32, z_dim, bias=True)
      self.batchnorm1 = nn.BatchNorm1d(z_dim)
      self.batchnorm2 = nn.BatchNorm1d(z_dim)

      self.dense_dec2 = nn.Linear(z_dim, int(HEIGHT/8)*int(WIDTH/8)*32)
      self.conv2dt_dec1 =  nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, bias=True)
      self.conv2dt_dec2 =  nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=True)
      self.conv2dt_dec3 =  nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, bias=True)

      self.dropout = nn.Dropout(0.25)


    def _encoder(self, x):  # Encodeでは、zの期待値と分散を作る
      x = F.relu(self.conv2d_enc1(x))
      x = F.relu(self.conv2d_enc2(x))
      x = F.relu(self.conv2d_enc3(x))
      x = self.flatten_enc1(x)
      x = self.dense_enc1(x)
      z = self.batchnorm1(x)
      z = F.sigmoid(z)
      z = self.batchnorm2(z)
      return(z)


    def _decoder(self, z):  # zからoutputのxを作る
      x = F.relu(self.dense_dec2(z))

      x = x.reshape(-1, 32, int(HEIGHT/8),int(WIDTH/8))
      x = F.relu(self.conv2dt_dec1(x))
      x = F.relu(self.conv2dt_dec2(x))
      x = F.sigmoid(self.conv2dt_dec3(x))
      #print(x.shape)
      return(x)


    def forward(self, x):
      z = self._encoder(x)  # inputをzに変換
      x = self._decoder(z)  # zをoutputに変換
      return(x, z)


    def loss(self, x, target):
      z = self._encoder(x)
      y = self._decoder(z)
      return(torch.mean((target-y)**2))
      

# カードベクトル化・引き当てを行うインスタンス
class CardClassifier:
    def __init__(self, ae_model_path, dictionary_path):
        # model load
        self.ae_model = torch.load(ae_model_path, map_location=torch.device('cpu'))
        self.ae_model.eval()
        # ベクトル辞書を読み込み
        print("loading vector dictionary...")
        with open(dictionary_path, "r") as f:
            data = json.load(f)
            self.filenames = data["filenames"]
            self.vectors = np.array(data["vectors"])

        # faiss 学習（インデックスを作成）
        print("training faiss...")
        self.index = faiss.IndexFlatL2(Z_DIM)
        self.index.add(self.vectors)
        
        
    def vectorize_image(self, target_image_path):
        image = cv2.imread(target_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
        v = self.trimmed_card_encode(image, self.ae_model)
        return(v)
        
    
    def search_index(self, v):
        D, I = self.index.search(v.reshape(1, Z_DIM), 5)
        for j in range(1):
            #print(f"候補{j}")
            print(self.filenames[I[0][j]], D[0][j])
            
        
    # トリムされたカードをエンコードする関数
    @staticmethod
    def trimmed_card_encode(cardimage: np.array, ae_model) -> np.array:
        image = cv2.resize(cardimage, (100, 140))
        image = image[0:80, 0:96, :]  # 画像の上半分、つまり絵柄の部分
        image = image_norm(image)  # 画像の標準化
        image = np.transpose(image, [2, 0, 1]).reshape(1, 3, 80, 96)
        image_for_torch = torch.from_numpy(image.astype(np.float32)).clone()
        image_for_torch = image_for_torch.to("cpu")
        z = ae_model(image_for_torch)[1][0].to('cpu').detach().numpy().copy()
        return(z)
