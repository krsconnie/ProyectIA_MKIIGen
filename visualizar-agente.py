import retro
import pickle
import neat
import numpy as np
import cv2
import random

def preprocess(obs):
    image = obs[0]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 84))
    image = image.astype(np.float32) / 255.0
    return image.flatten()

def jugar_con_agente(config_file, genoma_file):
    # Cargar configuración y agente
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    with open(genoma_file, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Crear entorno con renderizado
    env = retro.make(game="MortalKombatII-Genesis", state="Level1.LiuKangVsJax",
                     render_mode="human")
    obs = env.reset()[0]

    # Ruido inicial para romper simetrías
    for _ in range(random.randint(30, 120)):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs = env.reset()[0]


    total_reward = 0
    frame_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs_processed = preprocess(obs)
        output = net.activate(obs_processed)
        action = [1 if o > 0.5 else 0 for o in output]

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        frame_count += 1

    env.close()
    print(f"\n Partida terminada | Recompensa total: {total_reward} | Cuadros jugados: {frame_count}")

if __name__ == "__main__":
    jugar_con_agente("config-neat", "mejor_agente.pkl")
