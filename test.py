import retro
import pickle
import neat
import numpy as np
import cv2
import math
import time
import random
import lifes_laws
import pygame

# Mapeo de teclas a botones del emulador
KEY_TO_BUTTON = {
    pygame.K_w: 'UP',
    pygame.K_s: 'DOWN',
    pygame.K_a: 'LEFT',
    pygame.K_d: 'RIGHT',
    pygame.K_j: 'A',
    pygame.K_k: 'B',
    pygame.K_l: 'C',
    pygame.K_u: 'X',
    pygame.K_i: 'Y',
    pygame.K_o: 'Z',
    pygame.K_RETURN: 'START'
}

"""
Juegas contra un bot al MKII
"""
def jugar_humano(mapa=None):
    env = retro.make(game='MortalKombatII-Genesis', state=mapa, players=1, render_mode='human')
    obs = env.reset()[0]

    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Jugador Humano - Mortal Kombat II")
    clock = pygame.time.Clock()

    action = [0] * len(env.buttons)
    frame_count = 0
    done = False
    max_frames = 4500
    last_enemy_health = 120
    last_player_health = 120
    fitness = 0

    print("🎮 Controles: W/A/S/D para moverte | J/K/L para golpear | ENTER para Start")

    while not done:
        keys = pygame.key.get_pressed()
        action = [0] * len(env.buttons)

        for key_code, button in KEY_TO_BUTTON.items():
            if keys[key_code]:
                idx = env.buttons.index(button)
                action[idx] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                done = True

        _, _, terminated, truncated, info = env.step(action)
        done = done or terminated or truncated
        
        enemy_health = info.get("enemy_health", last_enemy_health)
        player_health = info.get("health", last_player_health)

        enemy_damage = max(0, last_enemy_health - enemy_health)
        self_damage = max(0, last_player_health - player_health)


        ##################################################################################
        ################################# Fitness adjust #################################
        ##################################################################################

        # Aqui va el fitness que quieras ajustar para tener referencia

        ##################################################################################
        ##################################################################################
        ##################################################################################
        

        last_enemy_health = enemy_health
        last_player_health = player_health
        frame_count += 1
        if frame_count > max_frames:
            fitness = -1000
            break
        clock.tick(60)


    print(f"\n🏁 Partida terminada. Fitness total: {fitness}, Cuadros jugados: {frame_count}")
    env.close()
    pygame.quit()

"""
El agente juega contra un bot al MKII
"""
def jugar_agente(genoma_file,  mapa=None, config_file="config-neat"):
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

    env = retro.make(game="MortalKombatII-Genesis",state=mapa , render_mode="human")
    obs, info = env.reset()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    INDICES_BOTONES = [env.buttons.index(b) for b in lifes_laws.BOTONES_USADOS]
    action = [0] * len(env.buttons)

    fitness = 0
    frame_count = 0
    terminated = False
    truncated = False
    last_enemy_health = 120
    last_player_health = 120
    max_frames = 4500


    while not (terminated or truncated) and frame_count < max_frames:
        max_x_pos = 320.0
        max_y_pos = 224.0
        pos_data = [
            info.get("x_position", 0) / max_x_pos,
            info.get("y_position", 0) / max_y_pos,
            info.get("enemy_x_position", 0) / max_x_pos,
            info.get("enemy_y_position", 0) / max_y_pos
        ]
        obs_processed = lifes_laws.tools.preprocess(obs)
        input_data = np.concatenate([obs_processed, pos_data])
        output = net.activate(input_data)

        action = [0] * len(env.buttons)
        for i, idx in enumerate(INDICES_BOTONES):
            action[idx] = 1 if output[i] > 0.5 else 0
        obs, _, terminated, truncated, info = env.step(action)

        enemy_health = info.get("enemy_health", last_enemy_health)
        player_health = info.get("health", last_player_health)

        enemy_damage = max(0, last_enemy_health - enemy_health)
        self_damage = max(0, last_player_health - player_health)

        ##################################################################################
        ################################# Fitness adjust #################################
        ##################################################################################

        # Aqui va el fitness que quieras ajustar para tener referencia

        ##################################################################################
        ##################################################################################
        ##################################################################################
        


        last_enemy_health = enemy_health
        last_player_health = player_health
        frame_count += 1

    env.close()
    print(f"\n🎮 Partida terminada | Recompensa total: {fitness} | Cuadros jugados: {frame_count}")
