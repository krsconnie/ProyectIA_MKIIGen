import retro
import pickle
import neat
import numpy as np
import cv2
import math
import time
import random

BOTONES_USADOS = ["A", "B", "C", 'START', "UP", "DOWN", "LEFT", "RIGHT"]

def preprocess(obs):
    image = obs[0]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (84, 84))
    image = image.astype(np.float32) / 255.0
    return image.flatten()

def fitness_funcion(agente_info, enemy_info):
    step_fitness = 0

    if enemy_info != 0:
        step_fitness += enemy_info
    
    if agente_info != 0:
        step_fitness -= agente_info

    return step_fitness

def jugar_con_agente(config_file, genoma_file, mapa=None):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    with open(genoma_file, "rb") as f:
        genome = pickle.load(f)

    print(f"El genoma con ID {genome.key}")
    if mapa is None:
        enemigo = f"VeryEasy.LiuKang-{random.randint(2, 15):02d}"
    else :
        enemigo = f"VeryEasy.LiuKang-{mapa:02d}"

    print(enemigo)

    env = retro.make(game="MortalKombatII-Genesis",state=enemigo , render_mode="human")
    obs = env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    INDICES_BOTONES = [env.buttons.index(b) for b in BOTONES_USADOS]
    action = [0] * len(env.buttons)

    total_reward = 0
    frame_count = 0
    terminated = False
    truncated = False
    last_enemy_health = 120
    last_player_health = 120
    max_frames = 10000

    for _ in range(random.randint(5, 30)):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()

    while not (terminated or truncated) and frame_count < max_frames:
        obs_processed = preprocess(obs)
        output = net.activate(obs_processed)

        action = [0] * len(env.buttons)
        for i, idx in enumerate(INDICES_BOTONES):
            action[idx] = 1 if output[i] > 0.5 else 0
        obs, reward, terminated, truncated, info = env.step(action)

        enemy_health = info.get("enemy_health", last_enemy_health)
        player_health = info.get("health", last_player_health)

        enemy_damage = max(0, last_enemy_health - enemy_health)
        self_damage = max(0, last_player_health - player_health)

        #print(f"Daño recibido: {self_damage}, Daño inflingido: {enemy_damage}")

        total_reward += fitness_funcion(self_damage, enemy_damage)
        last_enemy_health = enemy_health
        last_player_health = player_health
        frame_count += 1

    env.close()
    print(f"\n🎮 Partida terminada | Recompensa total: {total_reward} | Cuadros jugados: {frame_count}")

if __name__ == "__main__":
    
    # jugar_con_agente("config-neat", "mejor_agente.pkl", 2)
    # jugar_con_agente("config-neat", "mejor_agente.pkl", 3)
    # jugar_con_agente("config-neat", "mejor_agente.pkl", 6)
    # jugar_con_agente("config-neat", "mejor_agente.pkl", 11)
    for i in range(2, 16):
        jugar_con_agente("config-neat", "mejor_agente_v2.pkl", i)
