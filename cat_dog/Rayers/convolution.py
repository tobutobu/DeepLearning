import os, sys
import numpy as np
sys.path.append(os.pardir)  # パスに親ディレクトリ
from common.util import im2col, col2im


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        """"" Convolutionレイヤー

        Args:
            W (numpy.ndarray): フィルター(重み)、形状は(FN,C,FH,FW)
            b (numpy.ndarray): バイアス、形状は(FN)
            stride (int, optional): ストライド、デフォルトは1
            pad (int, optional): パディング、デフォルトは0
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.dW = None  # 重みの微分値
        self.db = None  # バイアスの微分値

        self.x = None   # 逆伝播で必要になる、順伝播時の入力
        self.col_x = None   # 逆伝播で必要になる、順伝播時の入力のcol展開結果
        self.col_W = None   # 逆伝播で必要になる、順伝播時のフィルターのcol展開結果

    def forward(self, x):
        """順伝播


        Args:
            x (numpy.ndarray): 入力、形状は(N,C,H,W)

        Returns:
            numpy.ndarray:　出力、形状は(N,FN,OH,OW)
        """
        FN, C, FH, FW = self.W.shape    # FN:フィルターの数、　C:チャンネルの数、　FH:フィルターの高さ、　FW:幅
        N, x_C, H, W = x.shape  # N: バッチサイズ、　x_C:チャネル数、　H:入力データの高さ、　W:幅
        assert C == x_C, f'チャンネル数の不一致！[C]{C}, [x_C]{x_C}'

        # 出力のサイズ算出
        assert (H + 2*self.pad - FH) % self.stride == 0, 'OHが割り切れない'
        assert (W + 2*self.pad - FH) % self.stride == 0, 'OWが割り切れない'
        OH = int((H + 2*self.pad - FH) / self.stride + 1)
        OW = int((W + 2*self.pad - FW) / self.stride + 1)

        # 入力データを展開
        # (N, C, H, W) →　(N * OH * OW, C * FH * FW)
        col_x = im2col(x, FH, FW, self.stride, self.pad)

        # フィルターを展開
        # (FN, C, FH, FW) →　(C * FH * FW, FN)
        col_W = self.W.reshape(FN, -1).T

        # 出力の算出
        # (N * OH * OW, C * FH * FW) ・ (C * FH * FW, FN) → (N * OH * OW, FW)
        out = np.dot(col_x, col_W) + self.b

        # 結果の整形
        # (N * OH * OW, FN) → (N, OH, OW, FN) →　(N, FN, OH, OW)
        out = out.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)

        # 逆伝播のために保存
        self.x = x
        self.col_x = col_x
        self.col_W = col_W

        return out

    def backward(self, dout):
        """逆伝播


        Args:
            dout (numpy.ndarray): 右の層から伝わってくる微分値、形状は(N, FN, OH, OW)

        Returns:
            numpy.ndarray: 微分値（勾配）、　形状は(N, C, H, W)
        """
        FN, C, FH, FW = self.W.shape    # 微分値の形状はWと同じ

        # 右の層からの微分値を展開
        # (N, FN, OH, OW) →　(N, OH, OW, FN) →　(N * OH * OW, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 微分値算出
        dcol_x = np.dot(dout, self.col_W.T)     # → (N * OH * OW, C * FH * FW)
        self.dW = np.dot(self.col_x.T, dout)    # →　(C * FH * FW, FN)
        self.db = np.sum(dout, axis=0)          # →　(FN)

        # フィルター（重み）の微分値の整形
        # (C * FH * FW, FN) →　(FN, C * FH * FW) →　(FN, C, FH, FW)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 結果（勾配）の整形
        # (N * OH * OW, C * FH * FW) →　(N, C, H, W)
        dx = col2im(dcol_x, self.x.shape, FH, FW, self.stride, self.pad)

        return dx