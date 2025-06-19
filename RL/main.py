import numpy as np
from mk_env import MortalKombatEnv

# Crear el entorno
env = MortalKombatEnv()

# Reiniciar el entorno antes de empezar
obs = env.reset()

# Variables para seguir la salud anterior (inicial)
p1_health_anterior = 120
p2_health_anterior = 120

# Loop de simulación
for i in range(5000):
    # Acciones aleatorias para el jugador 1
    action_p1 = env.action_space.sample()

    # Paso en el entorno
    obs, reward, terminated, truncated, info = env.step(action_p1)
    done = terminated or truncated

    # Extraer salud actual desde info
    p1_health_actual = info.get('health', p1_health_anterior)
    p2_health_actual = info.get('enemy_health', p2_health_anterior)

    # Calcular recompensa individual como daño hecho al enemigo
    reward_p1 = p2_health_anterior - p2_health_actual

    # Mostrar la info
    print(f"Paso {i}: Recompensa P1: {reward_p1}, Salud P1: {p1_health_actual}, Salud P2: {p2_health_actual}")

    # Actualizar salud anterior
    p1_health_anterior = p1_health_actual
    p2_health_anterior = p2_health_actual

    if done:
        print("Episodio terminado.")
        break

env.close()
