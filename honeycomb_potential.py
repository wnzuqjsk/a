import numpy as np
import matplotlib.pyplot as plt

# 定数とパラメータの設定
V_0 = 1.5
Delta = 0
a_0 = 1.0  # 格子定数
delta = np.array([1, 0])

# b_iの定義
def b_i(i):
    angle = 2 * np.pi * i / 3
    return (4 * np.pi / (3 * a_0)) * np.array([np.cos(angle), np.sin(angle)])

# 座標範囲の設定
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)
V_r = np.zeros_like(X, dtype=complex)

# ポテンシャルの計算
for i in range(1, 7):
    bi = b_i(i)
    V_r += (V_0 + Delta / 9 * np.exp(-1j * np.dot(b_i(i), delta))) * np.exp(1j * (bi[0] * X + bi[1] * Y))

# ポテンシャルの実数部を取り出す
V_r = V_r.real

# プロット
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, V_r, levels=50, cmap='plasma')  # 'plasma'カラーマップを使用
plt.colorbar(label='Potential')
plt.title('Honeycomb Lattice Potential')
plt.xlabel('x')
plt.ylabel('y')
plt.show()