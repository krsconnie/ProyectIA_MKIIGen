import numpy as np
import matplotlib.pyplot as plt
from mk_env import MortalKombatEnv
from stable_baselines3 import PPO

def evaluar_agente(env, model, episodios=7, repeticiones=10):
    todas_recompensas = []
    for rep in range(repeticiones):
        recompensas_totales = []
        for ep in range(episodios):
            obs, _ = env.reset()
            done = False
            recompensa_ep = 0
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                recompensa_ep += reward
            recompensas_totales.append(recompensa_ep)
            #print(f"[Modelo Run {rep+1}] Episodio {ep+1}: Recompensa = {recompensa_ep}")
        todas_recompensas.append(recompensas_totales)
    return todas_recompensas

def evaluar_random(env, episodios=7, repeticiones=10):
    todas_recompensas = []
    for rep in range(repeticiones):
        recompensas_totales = []
        for ep in range(episodios):
            obs, _ = env.reset()
            done = False
            recompensa_ep = 0
            while not done:
                action = env.action_space.sample()
                obs, reward, done, _, _ = env.step(action)
                recompensa_ep += reward
            recompensas_totales.append(recompensa_ep)
            #print(f"[Random Run {rep+1}] Episodio {ep+1}: Recompensa = {recompensa_ep}")
        todas_recompensas.append(recompensas_totales)
    return todas_recompensas

def graficar_comparacion(todas_recompensas_model, todas_recompensas_random):
    import numpy as np
    import matplotlib.pyplot as plt

    data_model = np.array(todas_recompensas_model)
    data_random = np.array(todas_recompensas_random)

    plt.figure(figsize=(6,5))
    # Usamos posiciones manuales más cercanas
    posiciones = [1, 1.3]

    box = plt.boxplot([data_model.flatten(), data_random.flatten()], 
                      patch_artist=True,
                      positions=posiciones,
                      widths=0.2,  # cajas más delgadas
                      labels=["Modelo PPO", "Agente Random"])

    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    promedio_model = np.mean(data_model)
    promedio_random = np.mean(data_random)
    plt.scatter([posiciones[0]], [promedio_model], color='blue', label=f'Promedio Modelo: {promedio_model:.2f}', zorder=10)
    plt.scatter([posiciones[1]], [promedio_random], color='green', label=f'Promedio Random: {promedio_random:.2f}', zorder=10)

    plt.title("Recompensas totales entre Modelo PPO y Agente Random")
    plt.ylabel("Recompensa total por episodio")
    plt.grid(axis='y')

    plt.legend()
    plt.tight_layout()
    plt.savefig('comparacion_recompensas_mejorada.png')
    plt.close()
    print("Gráfico guardado: comparacion_recompensas_mejorada.png")



if __name__ == "__main__":
    env = MortalKombatEnv()
    model_path = "./RL/ppo_mk_model"
    model = PPO.load(model_path)

    episodios = 7
    repeticiones = 10

    todas_recompensas_model = evaluar_agente(env, model, episodios=episodios, repeticiones=repeticiones)
    todas_recompensas_random = evaluar_random(env, episodios=episodios, repeticiones=repeticiones)

    graficar_comparacion(todas_recompensas_model, todas_recompensas_random)

    env.close()
