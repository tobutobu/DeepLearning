# coding: utf-8
import numpy as np


class AdaGrad:

    def __init__(self, lr=0.01):
        """AdaGradによるパラメーターの最適化

        Args:
            lr (float, optional): 学習係数、デフォルトは0.01。
        """
        self.lr = lr
        self.h = None   # これまでの勾配の2乗和

    def update(self, params, grads):
        """パラメーター更新

        Args:
            params (dict): 更新対象のパラメーターの辞書、keyは'W1'、'b1'など。
            grads (dict): paramsに対応する勾配の辞書
        """

        # hの初期化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 更新
        for key in params.keys():

            # hの更新
            self.h[key] += grads[key] ** 2

            # パラメーター更新、最後の1e-7は0除算回避
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)