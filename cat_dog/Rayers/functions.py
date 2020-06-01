# coding: utf-8
import numpy as np


def softmax(x):
    """ソフトマックス関数

    Args:
        x (numpy.ndarray): 入力

    Returns:
        numpy.ndarray: 出力
    """
    # バッチ処理の場合xは(バッチの数, 10)の2次元配列になる。
    # この場合、ブロードキャストを使ってうまく画像ごとに計算する必要がある。
    # ここでは1次元でも2次元でも共通化できるようnp.max()やnp.sum()はaxis=-1で算出し、
    # そのままブロードキャストできるようkeepdims=Trueで次元を維持する。
    c = np.max(x, axis=-1, keepdims=True)
    exp_a = np.exp(x - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    """交差エントロピー誤差の算出

    Args:
        y (numpy.ndarray): ニューラルネットワークの出力
        t (numpy.ndarray): 正解のラベル

    Returns:
        float: 交差エントロピー誤差
    """

    # データ1つ場合は形状を整形（1データ1行にする）
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 誤差を算出してバッチ数で正規化
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def conv_output_size(input_size, filter_size, pad, stride):
    """畳み込み層の出力サイズ算出

    Args:
        input_size (int): 入力の1辺のサイズ（縦横は同値の前提）
        filter_size (int): フィルターの1辺のサイズ（縦横は同値の前提）
        pad (int): パディングのサイズ（縦横は同値の前提）
        stride (int): ストライド幅（縦横は同値の前提）

    Returns:
        int: 出力の1辺のサイズ
    """
    assert (input_size + 2 * pad - filter_size) \
        % stride == 0, '畳み込み層の出力サイズが割り切れない！'
    return int((input_size + 2 * pad - filter_size) / stride + 1)


def pool_output_size(input_size, pool_size, stride):
    """プーリング層の出力サイズ算出

    Args:
        input_size (int): 入力の1辺のサイズ（縦横は同値の前提）
        pool_size (int): プーリングのウインドウサイズ（縦横は同値の前提）
        stride (int): ストライド幅（縦横は同値の前提）

    Returns:
        int: 出力の1辺のサイズ
    """
    assert (input_size - pool_size) % stride == 0, f'プーリング層の出力サイズが割り切れない！[input_size]{input_size}, [pool_size]{pool_size}, [stride]{stride}'
    return int((input_size - pool_size) / stride + 1)