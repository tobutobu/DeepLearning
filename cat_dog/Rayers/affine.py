import numpy as np


class Affine:

    def __init__(self, W, b):
        """Affineレイヤー


        Args:
            W (numpy.ndarray): 重み
            b (numpy.ndarray): バイアス
        """
        self.W = W                      # 重み
        self.b = b                      # バイアス
        self.x = None                   # 入力（2次元化後）
        self.dW = None                  # 重みの微分値
        self.db = None                  # バイアスの微分値
        self.original_x_shape = None    # 元の入力の形状（3次元以上の入力時用）

    def forward(self, x):
        """順伝播

        Args:
            x (numpy.ndarray): 入力

        Returns:
            numpy.ndarray: 出力
        """
        # 3次元以上（テンソル）の入力を2次元化
        self.original_x_shape = x.shape  # 形状を保存、逆伝播で戻す必要があるので
        x = x.reshape(x.shape[0], -1)
        self.x = x

        # 出力を算出
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (numpy.ndarray): 右の層から伝わってくる微分値

        Returns:
            numpy.ndarray: 微分値
        """
        # 微分値算出
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 元の形状に戻す
        dx = dx.reshape(*self.original_x_shape)
        return dx