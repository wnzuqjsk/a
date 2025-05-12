import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# 定数の設定
a_0 = 1.0
h_bar = 1.0
V_0 = 1.5
m = 1.5
Delta = 1
b_1 = -2 * np.pi / (3 * a_0)
δ = 1
k_points = 200  # サンプルポイントの数

# 波数ベクトルの範囲設定
k_values = np.linspace(-2 * np.pi / a_0, 2 * np.pi / a_0, k_points)

# 関数の定義
g = 1
w_c = 1
m_eff = m * (1 + (g / w_c) ** 2)

def lambda_k(h_bar, k, m_eff):
    return (h_bar ** 2 * k ** 2) / (2 * m_eff)

def V(b):
    return V_0 + Delta / 9 * np.exp(-1j * b)

# 各 k に対して固有値を計算
eigenvalues_list = []
num_bands = 20  # バンド数（行列のサイズ）を偶数に変更

for k in k_values:
    H = np.zeros((num_bands, num_bands), dtype=complex)
    center_band = (num_bands - 1) / 2 if num_bands % 2 != 0 else num_bands / 2 - 0.5
    for i in range(num_bands):
        for j in range(num_bands):
            if i == j:
                H[i, j] = lambda_k(h_bar, k + (i - center_band) * b_1, m_eff) + V(0)
            else:
                H[i, j] = V((i - j) * b_1)
    eigenvalues, _ = eigh(H)
    eigenvalues_list.append(eigenvalues.real)

# 固有値のリストを配列に変換
eigenvalues_array = np.array(eigenvalues_list)

# 固有値のプロット
plt.figure()
for i in range(eigenvalues_array.shape[1]):
    plt.plot(k_values, eigenvalues_array[:, i], label=f'Band {i+1}')
plt.xlabel('k')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues vs k')
plt.legend()
plt.show()
