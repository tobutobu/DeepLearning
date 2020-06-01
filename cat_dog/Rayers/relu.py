# coding: utf-8


class ReLU:
    def __init__(self):
        """ReLUレイヤー
        """
        self.mask = None

    def forward(self, x):
        """順伝播

        Args:
            x (numpy.ndarray): 入力

        Returns:
            numpy.ndarray: 出力
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (numpy.ndarray): 右の層から伝わってくる微分値

        Returns:
            numpy.ndarray: 微分値
        """
        dout[self.mask] = 0
        dx = dout

        return dx