import sys
sys.path.append('./RL')
import retro
import time

env = retro.make(game='MortalKombatII-Genesis', players=1, use_restricted_actions=retro.Actions.DISCRETE)
obs = env.reset()

n_acciones = env.action_space.n
print(f"Total de acciones posibles (DISCRETE): {n_acciones}")

for i in range(n_acciones):
    print(f"\n=== Acción {i} ===")
    try:
        meaning = env.get_action_meaning(i)
        print(f"Botones activos: {meaning}")
    except Exception as e:
        print(f"No se pudo obtener meaning para la acción {i}: {e}")

    # Ejecutar la acción durante un segundo
    for _ in range(150):
        obs, reward, terminated, truncated, info = env.step(0)
        env.render()
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(i)
        env.render()
        #print(f"Recompensa: {reward} | Info: {info}")

    env.reset()

env.close()
