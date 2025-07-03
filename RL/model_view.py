from stable_baselines3 import PPO
from mk_env import MortalKombatEnv  # Asegúrate de que tu entorno está bien importado
import time

# === Parámetros ===
model_path = "ppo_mk_model"  # Cambia por la ruta a tu modelo entrenado
max_steps = 10_000                     # Número de pasos que quieres ver
pause_time = 0.0                   # Tiempo entre frames, para ver mejor el render

# === Carga el entorno y el modelo ===
env = MortalKombatEnv()
model = PPO.load(model_path)
obs, _ = env.reset()
print("Modelo Cargado")

# === Loop de visualización ===
for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    #print("Inicio de step")
    obs, reward, done, _, info = env.step(action)
    env.render()
    #print("Step hecho")
    time.sleep(pause_time)

    # Si termina el episodio, reseteamos
    if done:
        obs, _ = env.reset()

env.close()
