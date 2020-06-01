import os, sys
import numpy as np
sys.path.append(os.pardir)
from common.util import im2col, col2im


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        """Poolingレイヤー


        Args:
            pool_h (int): プーリング領域の高さ
            pool_w (int): プーリング領域の幅
            stride (int, optional): ストライド、デフォルトは１
            pad (int, optional): パディング、　デフォルトは0
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None       # 逆伝播で必要になる、順伝播時の入力
        self.arg_max = None     # 逆伝播で必要になる、順伝播に採用したcol_x各行の位置

    def forward(self, x):
        """順伝播


         Args:
             x (numpy.ndarray): 入力、形状は(N,C,H,W)

         Returns:
             numpy.ndarray:　出力、形状は(N,C,OH,OW)
         """
        N, C, H, W = x.shape  # N:データ数、C:チャンネル数、H:高さ、W:幅

        # 出力のサイズ算出
        assert (H - self.pool_h) % self.stride == 0, f'OHが割り切れない！[H]{H}, [pool_h]{self.pool_h}, [stride]{self.stride}'
        assert (W - self.pool_w) % self.stride == 0, f'OWが割り切れない！[W]{W}, [pool_w]{self.pool_w}, [stride]{self.stride}'
        OH = int((H - self.pool_h) / self.stride + 1)
        OW = int((W - self.pool_w) / self.stride + 1)

        # 入力データを展開、整形
        # (N, C, H, W) → (N * OH * OW, C * PH * PW)
        col_x = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # (N * OH * OW, C * PH * PW) → (N * OH * OW * C, PH * PW)
        col_x = col_x.reshape(-1, self.pool_h * self.pool_w)

        # 出力を算出
        # (N * OH * OW * C, PH * PW) → (N * OH * OW * C)
        out = np.max(col_x, axis=1)

        # 結果の整形
        # (N * OH * OW * C) → (N, OH, OW, C) → (N, C, OH, OW)
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        # 逆伝播のために保存
        self.x = x
        self.arg_max = np.argmax(col_x, axis=1)  # col_x各行の最大値の位置（インデックス）

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (numpy.ndarray): 右の層から伝わってくる微分値、形状は(N, C, OH, OW)。

        Returns:
            numpy.ndarray: 微分値（勾配）、形状は(N, C, H, W)。
        """
        # 右の層からの微分値を整形
        # (N, C, OH, OW) → (N, OH, OW, C)
        dout = dout.transpose(0, 2, 3, 1)

        # 結果の微分値用のcolを0で初期化
        # (N * OH * OW * C, PH * PW)
        pool_size = self.pool_h * self.pool_w
        dcol_x = np.zeros((dout.size, pool_size))

        # 順伝播時に最大値として採用された位置にだけ、doutの微分値（＝doutまんま）をセット
        # 順伝播時に採用されなかった値の位置は初期化時の0のまま
        # （ReLUでxが0より大きい場合およびxが0以下の場合の処理と同じ）
        assert dout.size == self.arg_max.size, '順伝搬時のcol_xの行数と合わない'
        dcol_x[np.arange(self.arg_max.size), self.arg_max.flatten()] = \
            dout.flatten()

        # 結果の微分値の整形1
        # (N * OH * OW * C, PH * PW) → (N, OH, OW, C, PH * PW)
        dcol_x = dcol_x.reshape(dout.shape + (pool_size,))  # 最後の','は1要素のタプルを示す

        # 結果の微分値の整形2
        # (N, OH, OW, C, PH * PW) → (N * OH * OW, C * PH * PW)
        dcol_x = dcol_x.reshape(
            dcol_x.shape[0] * dcol_x.shape[1] * dcol_x.shape[2], -1
        )

        # 結果の微分値の整形3
        # (N * OH * OW, C * PH * PW) → (N, C, H, W)
        dx = col2im(
            dcol_x, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad
        )

        return dx