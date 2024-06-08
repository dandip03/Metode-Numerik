
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Impor data dari file CSV
file_path = 'Student_Performance.csv'  # ganti dengan path file CSV Anda
data = pd.read_csv(file_path)

# Ekstrak kolom yang dibutuhkan
TB = data['Sleep Hours']
NT = data['Performance Index']

# Reshape data
TB_reshaped = TB.values.reshape(-1, 1)
NT_reshaped = NT.values.reshape(-1, 1)

# 2. Implementasi Model
# Metode 1: Regresi Linear
linear_model = LinearRegression()
linear_model.fit(TB_reshaped, NT_reshaped)
NT_pred_linear = linear_model.predict(TB_reshaped)

rmse_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
# Plot hasil regresi linear
plt.scatter(TB, NT, color='red', label='Data')
plt.plot(TB, NT_pred_linear, color='violet', label='Regresi Linear')
plt.xlabel('Sleep Hours')
plt.ylabel('Performance Index')
plt.title('Regresi Linear: Sleep Hours vs Performance Index')
plt.legend()
plt.figtext(0.2, 0, f' RMSE (Regresi Eksponensial): {rmse_linear}', fontsize=10, ha='center')
plt.show()

# Hitung RMSE untuk Regresi Linear
print(f'RMSE (Regresi Linear): {rmse_linear}')

# Metode 3: Regresi Eksponensial
# Fungsi eksponensial
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Cari parameter yang cocok untuk model eksponensial
params, covariance = curve_fit(exp_func, TB, NT)
a, b = params
NT_pred_exp = exp_func(TB, a, b)

rmse_exp = np.sqrt(mean_squared_error(NT, NT_pred_exp))

# Plot hasil regresi eksponensial
plt.scatter(TB, NT, color='red', label='Data')
plt.plot(TB, NT_pred_exp, color='yellow', label='Regresi Eksponensial')
plt.xlabel('Sleep Hours')
plt.ylabel('Performance Index')
plt.title('Regresi Eksponensial: Sleep Hours vs Performance Index')
plt.legend()
plt.figtext(0., 0, f' RMSE (Regresi Eksponensial): {rmse_exp}', fontsize=10, ha='center')
plt.show()

# Hitung RMSE untuk Regresi Eksponensial
print(f'RMSE (Regresi Eksponensial): {rmse_exp}')



# Hasil Pengujian
hasil_pengujian = f"""
- RMSE untuk Regresi Linear: {rmse_linear}
- RMSE untuk Regresi Eksponensial: {rmse_exp}
"""

# Analisis Hasil
analisis_hasil = f"""
Dari hasil yang diperoleh, dapat dilihat bahwa {'regresi linear' if rmse_linear < rmse_exp else 'regresi eksponensial'} memiliki RMSE yang lebih kecil. Hal ini menunjukkan bahwa model {'linear' if rmse_linear < rmse_exp else 'eksponensial'} lebih baik dalam memprediksi nilai ujian berdasarkan waktu tidur siswa.
"""

# Cetak dokumentasi dan analisis
print(hasil_pengujian)
print(analisis_hasil)