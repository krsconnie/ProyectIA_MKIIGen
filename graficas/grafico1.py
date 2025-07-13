import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Leer CSV
df = pd.read_csv('historial_damage.csv')
x = df['NumGen']
y = df['Prom_damage']

# Crear más puntos para curva suave
x_new = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_new)

# Graficar
plt.figure(figsize=(8,5))
plt.plot(x, y, 'o', label='Original')
plt.plot(x_new, y_smooth, label='Spline Suavizado', color='green')
plt.title('Promedio de daño por generación')
plt.xlabel('Número de generación')
plt.ylabel('Promedio de daño')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
