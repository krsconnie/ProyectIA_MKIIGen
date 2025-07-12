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
import lifes_laws

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
        print(math.sqrt((info.get("enemy_x_position", 0) - info.get("x_position", 0))**2 + (info.get("enemy_y_position", 0) - info.get("y_position", 0))**2))
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

    action = [0] * len(lifes_laws.config.BOTONES_USADOS)

    fitness = 0
    frame_count = 0
    terminated = False
    truncated = False
    last_enemy_health = 120
    last_player_health = 120
    max_frames = 4500
    done = False

    while not done and frame_count < max_frames:

        max_x_pos = 320.0   # Máxima posicion X en el mapa
        max_y_pos = 224.0   # Máxima posicion Y en el mapa
        
        # pos_data contiene la ubicación de del agente y del enemigo
        pos_data = [
            info.get("x_position", 0) / max_x_pos,
            info.get("y_position", 0) / max_y_pos,
            info.get("enemy_x_position", 0) / max_x_pos,
            info.get("enemy_y_position", 0) / max_y_pos
        ]
        obs_processed = lifes_laws.tools.preprocess(obs)
        input_data = np.concatenate([obs_processed, pos_data]) # Juntamos la obs con pos_data
        output = net.activate(input_data)   # La activamos, para esto tuvimos que subir las neuronas input de 7056 a 7060

        for i in range(8):
            action[i] = 1 if output[i] > 0.5 else 0

        #if tuple(action) in exe_config.MOVIMIENTOS_BASE:
            #avg_fitness[i_escn] += 1
            #if exe_config.MOVIMIENTOS_BASE[tuple(action)] not  in acciones_utilizadas:
                #acciones_utilizadas.add(exe_config.MOVIMIENTOS_BASE[tuple(action)])
                #avg_fitness[i_escn] += 200
            #ultimas_acciones.append((exe_config.MOVIMIENTOS_BASE[tuple(action)], frame_count))


        # distancia = math.sqrt((info.get("enemy_x_position", 0) - info.get("x_position", 0))**2 + (info.get("enemy_y_position", 0) - info.get("y_position", 0))**2)

        # if 100 < distancia < 150:
        #     avg_fitness[i_escn] += 1
        # else:
        #     avg_fitness[i_escn] -= 1
        

        # Mantenemos la acción por la cantidad de frames indicada en FRAME_SKIP. Esto hace hasta (FRAME_SKIP - 1) veces más rapido el entrenamiento
        for _ in range(lifes_laws.config.FRAME_SKIP):
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # También aumentamos el contador por cada frame saltado
            frame_count += 1
            if done or frame_count >= max_frames: 
                break

            # 👇 Procesamiento de daño dentro del loop skip (opcional)
            enemy_health = info.get("enemy_health", last_enemy_health)
            player_health = info.get("health", last_player_health)

            enemy_damage = max(0, last_enemy_health - enemy_health)
            self_damage = max(0, last_player_health - player_health)
            
            #avg_fitness[i_escn] += 1
            fitness += enemy_damage
            fitness -= self_damage

            last_enemy_health = enemy_health
            last_player_health = player_health

            if info.get("rounds_won", 0) != 0 or info.get("enemy_rounds_won", 0) != 0:
                done = True
                break

    fitness += 120

    env.close()
    
    print(f"\n🎮 Partida terminada | Recompensa total: {fitness} | Cuadros jugados: {frame_count}")
