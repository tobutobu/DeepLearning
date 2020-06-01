# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # パスに親ディレクトリ追加
from Rayers.functions import softmax, cross_entropy_error


class SoftmaxWithLoss:
    def __init__(self):
        """Softmax-with-Lossレイヤー
        """
        self.loss = None    # 損失
        self.y = None       # softmaxの出力
        self.t = None       # 教師データ（one-hot vector）

    def forward(self, x, t):
        """順伝播

        Args:
            x (numpy.ndarray): 入力
            t (numpy.ndarray): 教師データ

        Returns:
            float: 交差エントロピー誤差
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (float, optional): 右の層から伝わってくる微分値。デフォルトは1。

        Returns:
            numpy.ndarray: 微分値
        """
        batch_size = self.t.shape[0]    # バッチの個数
        dx = (self.y - self.t) * (dout / batch_size)

        return dx