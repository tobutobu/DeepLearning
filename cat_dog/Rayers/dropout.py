# coding: utf-8
import numpy as np


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        """Dropoutレイヤー

        Args:
            dropout_ratio (float): 学習時のニューロンの消去割合、デフォルトは0.5。
        """
        self.dropout_ratio = dropout_ratio              # 学習時のニューロンの消去割合
        self.valid_ratio = 1.0 - self.dropout_ratio     # 学習時に生かしていた割合
        self.mask = None                                # 各ニューロンの消去有無を示すフラグの配列

    def forward(self, x, train_flg=True):
        """順伝播

        Args:
            x (numpy.ndarray): 入力
            train_flg (bool, optional): 学習中ならTrue、デフォルトはTrue。

        Returns:
            numpy.ndarray: 出力
        """
        if train_flg:
            # 学習時は消去するニューロンを決めるマスクを生成
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio

            # 出力を算出
            return x * self.mask

        else:
            # 認識時はニューロンは消去しないが、学習時の消去割合を加味した出力に調整する
            return x * self.valid_ratio

    def backward(self, dout):
        """逆伝播

        Args:
            dout (numpy.ndarray): 右の層から伝わってくる微分値

        Returns:
            numpy.ndarray: 微分値（勾配）
        """
        # 消去しなかったニューロンのみ右の層の微分値を逆伝播
        assert self.mask is not None, '順伝播なしに逆伝播が呼ばれた'
        return dout * self.mask