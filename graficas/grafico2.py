import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 1️⃣ Leer el CSV
df = pd.read_csv('historial2.csv')

# 2️⃣ Extraer las columnas
gen = df['Gen']
prom_fit = df['Prom_fit']
prom_pos = df['Prom_positive_fit']
prom_neg = df['Prom_negative_fit']

# 3️⃣ Crear puntos interpolados para suavizar
gen_smooth = np.linspace(gen.min(), gen.max(), 300)

# 4️⃣ Ajustar splines
spl_fit = make_interp_spline(gen, prom_fit, k=3)
spl_pos = make_interp_spline(gen, prom_pos, k=3)
spl_neg = make_interp_spline(gen, prom_neg, k=3)

# 5️⃣ Evaluar splines
fit_smooth = spl_fit(gen_smooth)
pos_smooth = spl_pos(gen_smooth)
neg_smooth = spl_neg(gen_smooth)

# 6️⃣ Graficar
plt.figure(figsize=(10,6))
plt.plot(gen_smooth, fit_smooth, label='Promedio Fitness (suavizado)')
plt.plot(gen_smooth, pos_smooth, label='Fitness Positivo (suavizado)')
plt.plot(gen_smooth, neg_smooth, label='Fitness Negativo (suavizado)')

plt.title('Evolución del Fitness por Generación (Suavizado)')
plt.xlabel('Generación')
plt.ylabel('Fitness promedio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
