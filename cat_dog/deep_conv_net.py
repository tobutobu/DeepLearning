# coding: utf-8
import numpy as np
import pickle
from Rayers.affine import Affine
from Rayers.convolution import Convolution
from Rayers.dropout import Dropout
from Rayers.functions import conv_output_size, pool_output_size
from Rayers.pooling import Pooling
from Rayers.relu import ReLU
from Rayers.softmax_with_loss import SoftmaxWithLoss


class DeepConvNet:

    def __init__(
        self, input_dim=(1, 28, 28),
        conv_param_1={
            'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_2={
            'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_3={
            'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_4={
            'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_5={
            'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_6={
            'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        hidden_size=50, output_size=10
    ):
        """ディープな畳み込みニューラルネットワーク

        Args:
            input_dim (tuple, optional): 入力データの形状、デフォルトは(1, 28, 28)。
            conv_param_1 (dict, optional): 畳み込み層1のハイパーパラメーター、
                デフォルトは{'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}。
            conv_param_2 (dict, optional): 畳み込み層2のハイパーパラメーター、
                デフォルトは{'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}。
            conv_param_3 (dict, optional): 畳み込み層3のハイパーパラメーター、
                デフォルトは{'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1}。
            conv_param_4 (dict, optional): 畳み込み層4のハイパーパラメーター、
                デフォルトは{'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1}。
            conv_param_5 (dict, optional): 畳み込み層5のハイパーパラメーター、
                デフォルトは{'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}。
            conv_param_6 (dict, optional): 畳み込み層6のハイパーパラメーター、
                デフォルトは{'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}。
            hidden_size (int, optional): 隠れ層のニューロンの数、デフォルトは50。
            output_size (int, optional): 出力層のニューロンの数、デフォルトは10。
        """
        assert input_dim[1] == input_dim[2], '入力データは高さと幅が同じ前提！'

        # パラメーターの初期化とレイヤー生成
        self.params = {}    # パラメーター
        self.layers = {}    # レイヤー（Python 3.7からは辞書の格納順が保持されるので、OrderedDictは不要）

        # 入力サイズ
        channel_num = input_dim[0]                          # 入力のチャンネル数
        input_size = input_dim[1]                           # 入力サイズ

        # [1] 畳み込み層#1 : パラメーター初期化とレイヤー生成
        filter_num, filter_size, pad, stride = list(conv_param_1.values())
        pre_node_num = channel_num * (filter_size ** 2)     # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W1', 'b1'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(filter_num, channel_num, filter_size, filter_size)
        )
        self.params[key_b] = np.zeros(filter_num)

        self.layers['Conv1'] = Convolution(
            self.params[key_w], self.params[key_b], stride, pad
        )

        # 次の層の入力サイズ算出
        channel_num = filter_num
        input_size = conv_output_size(input_size, filter_size, pad, stride)

        # [2] ReLU層#1 : レイヤー生成
        self.layers['ReLU1'] = ReLU()

        # [3] 畳み込み層#2 : パラメーター初期化とレイヤー生成
        filter_num, filter_size, pad, stride = list(conv_param_2.values())
        pre_node_num = channel_num * (filter_size ** 2)     # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W2', 'b2'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(filter_num, channel_num, filter_size, filter_size)
        )
        self.params[key_b] = np.zeros(filter_num)

        self.layers['Conv2'] = Convolution(
            self.params[key_w], self.params[key_b], stride, pad
        )

        # 次の層の入力サイズ算出
        channel_num = filter_num
        input_size = conv_output_size(input_size, filter_size, pad, stride)

        # [4] ReLU層#2 : レイヤー生成
        self.layers['ReLU2'] = ReLU()

        # [5] プーリング層#1 : レイヤー生成
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 次の層の入力サイズ算出
        input_size = pool_output_size(input_size, pool_size=2, stride=2)

        # [6] 畳み込み層#3 : パラメーター初期化とレイヤー生成
        filter_num, filter_size, pad, stride = list(conv_param_3.values())
        pre_node_num = channel_num * (filter_size ** 2)     # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W3', 'b3'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(filter_num, channel_num, filter_size, filter_size)
        )
        self.params[key_b] = np.zeros(filter_num)

        self.layers['Conv3'] = Convolution(
            self.params[key_w], self.params[key_b], stride, pad
        )

        # 次の層の入力サイズ算出
        channel_num = filter_num
        input_size = conv_output_size(input_size, filter_size, pad, stride)

        # [7] ReLU層#3 : レイヤー生成
        self.layers['ReLU3'] = ReLU()

        # [8] 畳み込み層#4 : パラメーター初期化とレイヤー生成
        filter_num, filter_size, pad, stride = list(conv_param_4.values())
        pre_node_num = channel_num * (filter_size ** 2)     # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W4', 'b4'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(filter_num, channel_num, filter_size, filter_size)
        )
        self.params[key_b] = np.zeros(filter_num)

        self.layers['Conv4'] = Convolution(
            self.params[key_w], self.params[key_b], stride, pad
        )

        # 次の層の入力サイズ算出
        channel_num = filter_num
        input_size = conv_output_size(input_size, filter_size, pad, stride)

        # [9] ReLU層#4 : レイヤー生成
        self.layers['ReLU4'] = ReLU()

        # [10] プーリング層#2 : レイヤー生成
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 次の層の入力サイズ算出
        input_size = pool_output_size(input_size, pool_size=2, stride=2)

        # [11] 畳み込み層#5 : パラメーター初期化とレイヤー生成
        filter_num, filter_size, pad, stride = list(conv_param_5.values())
        pre_node_num = channel_num * (filter_size ** 2)     # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W5', 'b5'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(filter_num, channel_num, filter_size, filter_size)
        )
        self.params[key_b] = np.zeros(filter_num)

        self.layers['Conv5'] = Convolution(
            self.params[key_w], self.params[key_b], stride, pad
        )

        # 次の層の入力サイズ算出
        channel_num = filter_num
        input_size = conv_output_size(input_size, filter_size, pad, stride)

        # [12] ReLU層#5 : レイヤー生成
        self.layers['ReLU5'] = ReLU()

        # [13] 畳み込み層#6 : パラメーター初期化とレイヤー生成
        filter_num, filter_size, pad, stride = list(conv_param_6.values())
        pre_node_num = channel_num * (filter_size ** 2)     # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W6', 'b6'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(filter_num, channel_num, filter_size, filter_size)
        )
        self.params[key_b] = np.zeros(filter_num)

        self.layers['Conv6'] = Convolution(
            self.params[key_w], self.params[key_b], stride, pad
        )

        # 次の層の入力サイズ算出
        channel_num = filter_num
        input_size = conv_output_size(input_size, filter_size, pad, stride)

        # [14] ReLU層#6 : レイヤー生成
        self.layers['ReLU6'] = ReLU()

        # [15] プーリング層#3 : レイヤー生成
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 次の層の入力サイズ算出
        input_size = pool_output_size(input_size, pool_size=2, stride=2)

        # [16] Affine層#1　: パラメーター初期化とレイヤー生成
        pre_node_num = channel_num * (input_size ** 2)      # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W7', 'b7'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(channel_num * (input_size ** 2), hidden_size)
        )
        self.params[key_b] = np.zeros(hidden_size)

        self.layers['Affine1'] = Affine(self.params[key_w], self.params[key_b])

        # 次の層の入力サイズ算出
        input_size = hidden_size

        # [17] ReLU層#7 : レイヤー生成
        self.layers['ReLU7'] = ReLU()

        # [18] Dropout層#１ : レイヤー生成
        self.layers['Drop1'] = Dropout(dropout_ratio=0.5)

        # [19] Affine層#2　: パラメーター初期化とレイヤー生成
        pre_node_num = input_size                           # 1ノードに対する前層の接続ノード数
        key_w, key_b = 'W8', 'b8'                           # 辞書格納時のkey
        self.params[key_w] = np.random.normal(
            scale=np.sqrt(2.0 / pre_node_num),              # Heの初期値の標準偏差
            size=(input_size, output_size)
        )
        self.params[key_b] = np.zeros(output_size)

        self.layers['Affine2'] = Affine(self.params[key_w], self.params[key_b])

        # [20] Dropout層#2 : レイヤー生成
        self.layers['Drop2'] = Dropout(dropout_ratio=0.5)

        # [21] Softmax層 : レイヤー生成
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        """ニューラルネットワークによる推論

        Args:
            x (numpy.ndarray): ニューラルネットワークへの入力
            train_flg (Boolean): 学習中ならTrue（Dropout層でニューロンの消去を実施）

        Returns:
            numpy.ndarray: ニューラルネットワークの出力
        """
        # レイヤーを順伝播
        for layer in self.layers.values():
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)  # Dropout層の場合は、学習中かどうかを伝える
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        """損失関数の値算出

        Args:
            x (numpy.ndarray): ニューラルネットワークへの入力
            t (numpy.ndarray): 正解のラベル

        Returns:
            float: 損失関数の値
        """
        # 推論
        y = self.predict(x, True)   # 損失は学習中しか算出しないので常にTrue

        # Softmax-with-Lossレイヤーの順伝播で算出
        loss = self.lastLayer.forward(y, t)

        return loss

    def accuracy(self, x, t, batch_size=100):
        """認識精度算出
        batch_sizeは算出時のバッチサイズ。一度に大量データを算出しようとすると
        im2colでメモリを食い過ぎてスラッシングが起きてしまい動かなくなるため、
        その回避のためのもの。

        Args:
            x (numpy.ndarray): ニューラルネットワークへの入力
            t (numpy.ndarray): 正解のラベル（one-hot）
            batch_size (int), optional): 算出時のバッチサイズ、デフォルトは100。

        Returns:
            float: 認識精度
        """
        # 分割数算出
        batch_num = max(int(x.shape[0] / batch_size), 1)

        # 分割
        x_list = np.array_split(x, batch_num, 0)
        t_list = np.array_split(t, batch_num, 0)

        # 分割した単位で処理
        correct_num = 0  # 正答数の合計
        for (sub_x, sub_t) in zip(x_list, t_list):
            assert sub_x.shape[0] == sub_t.shape[0], '分割境界がずれた？'
            y = self.predict(sub_x, False)  # 認識精度は学習中は算出しないので常にFalse
            y = np.argmax(y, axis=1)
            t = np.argmax(sub_t, axis=1)
            correct_num += np.sum(y == t)

        # 認識精度の算出
        return correct_num / x.shape[0]

    def gradient(self, x, t):
        """重みパラメーターに対する勾配を誤差逆伝播法で算出

         Args:
            x (numpy.ndarray): ニューラルネットワークへの入力
            t (numpy.ndarray): 正解のラベル

        Returns:
            dictionary: 勾配を格納した辞書
        """
        # 順伝播
        self.loss(x, t)     # 損失値算出のために順伝播する

        # 逆伝播
        dout = self.lastLayer.backward()
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        # 各レイヤーの微分値を取り出し
        grads = {}
        layer = self.layers['Conv1']
        grads['W1'], grads['b1'] = layer.dW, layer.db
        layer = self.layers['Conv2']
        grads['W2'], grads['b2'] = layer.dW, layer.db
        layer = self.layers['Conv3']
        grads['W3'], grads['b3'] = layer.dW, layer.db
        layer = self.layers['Conv4']
        grads['W4'], grads['b4'] = layer.dW, layer.db
        layer = self.layers['Conv5']
        grads['W5'], grads['b5'] = layer.dW, layer.db
        layer = self.layers['Conv6']
        grads['W6'], grads['b6'] = layer.dW, layer.db
        layer = self.layers['Affine1']
        grads['W7'], grads['b7'] = layer.dW, layer.db
        layer = self.layers['Affine2']
        grads['W8'], grads['b8'] = layer.dW, layer.db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate(('Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Affine1', 'Affine2')):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]