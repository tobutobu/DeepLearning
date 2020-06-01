# coding: utf-8
import numpy as np


class Adam:

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999):
        """Adamによるパラメーターの最適化

        Args:
            alpha (float, optional): 学習係数、デフォルトは0.001。
            beta1 (float, optional): Momentumにおける速度の過去と今の按分の係数、デフォルトは0.9。
            beta2 (float, optional): AdaGradにおける学習係数の過去と今の按分の係数、デフォルトは0.999。
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = None   # Momentumにおける速度
        self.v = None   # AdaGradにおける学習係数
        self.t = 0      # タイムステップ

    def update(self, params, grads):
        """パラメーター更新

        Args:
            params (dict): 更新対象のパラメーターの辞書、keyは'W1'、'b1'など。
            grads (dict): paramsに対応する勾配の辞書
        """
        # mとvの初期化
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        # 更新
        self.t += 1     # タイムステップ加算
        for key in params.keys():

            # mの更新、Momentumにおける速度の更新に相当
            # 過去と今の勾配を beta1 : 1 - beta1 で按分する
            self.m[key] = \
                self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # vの更新、AdaGradにおける学習係数の更新に相当
            # 過去と今の勾配を beta2 : 1 - beta2 で按分する
            self.v[key] = \
                self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # パラメーター更新のためのmとvの補正値算出
            hat_m = self.m[key] / (1.0 - self.beta1 ** self.t)
            hat_v = self.v[key] / (1.0 - self.beta2 ** self.t)

            # パラメーター更新、最後の1e-7は0除算回避
            params[key] -= self.alpha * hat_m / (np.sqrt(hat_v) + 1e-7)