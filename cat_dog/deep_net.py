# coding: utf-8
import os
import sys
import matplotlib.pylab as plt
import numpy as np
from Rayers.adam import Adam
from deep_conv_net import DeepConvNet
sys.path.append(os.pardir)  # パスに親ディレクトリ追加
from data.cat_dog_dataset import load_data


# dataの訓練データとテストデータ読み込み
(x_train, t_train), (x_test, t_test) = load_data(normalize=True, plot_show=False, one_hot_label=True)

# データの削除
x_train, t_train = x_train[:1600], t_train[:1600]
x_test, t_test = x_test[:400], t_test[:400]

# ハイパーパラメーター設定
iters_num = 960           # 更新回数
batch_size = 100            # バッチサイズ
adam_param_alpha = 0.0004    # Adamの学習係数
adam_param_beta1 = 0.9      # Adamのパラメーター
adam_param_beta2 = 0.999    # Adamのパラメーター

train_size = x_train.shape[0]  # 訓練データのサイズ
iter_per_epoch = max(int(train_size / batch_size), 1)    # 1エポック当たりの繰り返し数

# ディープな畳み込みニューラルネットワーク生成
network = DeepConvNet(
        input_dim=(3, 32, 32),
        conv_param_1={
            'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_2={
            'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_3={
            'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_4={
            'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_5={
            'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_6={
            'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        hidden_size=50, output_size=2
)

# オプティマイザー生成、Adamを使用
optimizer = Adam(adam_param_alpha, adam_param_beta1, adam_param_beta2)

# 学習前の認識精度の確認
train_acc = network.accuracy(x_train, t_train)
test_acc = network.accuracy(x_test, t_test)
train_loss_list = []            # 損失関数の値の推移の格納先
train_acc_list = [train_acc]    # 訓練データに対する認識精度の推移の格納先
test_acc_list = [test_acc]      # テストデータに対する認識精度の推移の格納先
print(f'学習前 [訓練データの認識精度]{train_acc:.4f} [テストデータの認識精度]{test_acc:.4f}')

# 学習開始
for i in range(iters_num):

    # ミニバッチ生成
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grads = network.gradient(x_batch, t_batch)

    # 重みパラメーター更新
    optimizer.update(network.params, grads)

    # 損失関数の値算出
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックごとに認識精度算出
    if (i + 1) % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 経過表示
        print(
            f'[エポック]{(i + 1) // iter_per_epoch:>2} '
            f'[更新数]{i + 1:>5} [損失関数の値]{loss:.4f} '
            f'[訓練データの認識精度]{train_acc:.4f} [テストデータの認識精度]{test_acc:.4f}'
        )

# パラメータの保存
network.save_params("params_2000.pkl")
print("Saved Network Parameters!")


# 損失関数の値の推移を描画
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.xlim(left=0)
plt.ylim(0, 2.5)
plt.show()

# 訓練データとテストデータの認識精度の推移を描画
x2 = np.arange(len(train_acc_list))
plt.plot(x2, train_acc_list, label='train acc')
plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.xlim(left=0)
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()