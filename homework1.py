import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data= np.genfromtxt("hist2.csv", delimiter=',')
x = data[:, 0]
y = data[:, 1]

plt.bar(x,y, width=0.082 ,color='k', alpha=0.5)
plt.xlim(0,4)
plt.title("energy spectrum")
plt.xlabel("energy(keV)")
plt.ylabel("count")
plt.grid()
plt.show()

x = data[:, 0]  # 첫 번째 열 (에너지)
y = data[:, 1]  # 두 번째 열 (세기)

# 두 개의 가우시안 함수 정의
def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
            a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))

# 가우시안 피팅
params, covariance = curve_fit(double_gaussian, x, y)

# 피팅된 데이터 계산
fitted_y = double_gaussian(x, *params)

# 피팅된 각각의 가우시안 분리
gaussian1 = params[0] * np.exp(-(x - params[1])**2 / (2 * params[2]**2))
gaussian2 = params[3] * np.exp(-(x - params[4])**2 / (2 * params[5]**2))

# 피팅 결과 시각화
plt.scatter(x, y, label='Measured Data', color='blue', s=10)
plt.plot(x, fitted_y, color='red',linestyle='--', label='Fitted Double Gaussian')
plt.plot(x, gaussian1, color='green', label='Gaussian 1')
plt.plot(x, gaussian2, color='orange', label='Gaussian 2')
plt.xlabel(' energy ')
plt.ylabel('count')
plt.title('double gaussian fitting')
plt.legend()
plt.show()

# 피팅 결과 출력
print("피팅된 파라미터:")
print(f" a1: {params[0]:.2f}, mu1: {params[1]:.2f}, sigma1: {params[2]:.2f}")
print(f" a2: {params[3]:.2f}, mu2: {params[4]:.2f}, sigma2: {params[5]:.2f}")

#면적구하기
area1 = params[0] * params[2] * np.sqrt(2 * np.pi)
area2 = params[3] * params[5] * np.sqrt(2 * np.pi)
print(f"Gaussian 1 면적: {area1:.2f}")
print(f"Gaussian 2 면적: {area2:.2f}")

#두 입자의 생성비
print(f"두 입자의 생성비 : {area1:.2f}, {area2:.2f}")