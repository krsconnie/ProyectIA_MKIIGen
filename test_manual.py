import retro
import pygame
import time
import math

def distancia_funcion(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def fitness_funcion(agente_info, enemy_info):
    step_fitness = 0
    ideal = 70
    margen = 10  # tolerancia entre 110 y 130

    distancia = distancia_funcion(agente_info[1], agente_info[2], enemy_info[1], enemy_info[2])
    delta = abs(distancia - ideal)

    print(distancia)

    if delta <= margen:
        pass
        #step_fitness += 0.3  # recompensa por estar a buena distancia
    else:
        penalizacion = (delta - margen) ** 2 * 0.001
        #step_fitness -= penalizacion

    if enemy_info[0] != 0:
        step_fitness += 2000  # hizo daño al enemigo

    if agente_info[0] != 0:
        step_fitness -= 2000  # recibió daño

    return step_fitness


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

def jugar_humano():
    env = retro.make(game='MortalKombatII-Genesis', players=1, render_mode='human')
    obs = env.reset()[0]

    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Jugador Humano - Mortal Kombat II")
    clock = pygame.time.Clock()

    action = [0] * len(env.buttons)
    total_reward = 0
    frame_count = 0
    done = False
    max_frames = 10000
    last_enemy_health = 120
    last_player_health = 120
    fitness = 0

    print("🎮 Controles: W/A/S/D para moverte | J/K/L para golpear | ENTER para Start")

    while not done and frame_count < max_frames:
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

        obs, reward, terminated, truncated, info = env.step(action)
        done = done or terminated or truncated
        
        enemy_health = info.get("enemy_health", last_enemy_health)
        player_health = info.get("health", last_player_health)

        enemy_damage = max(0, last_enemy_health - enemy_health)
        self_damage = max(0, last_player_health - player_health)
        self_x = info.get("x_position", 0)
        self_y = info.get("y_position", 0)
        enemy_x = info.get("enemy_x_position", 0)
        enemy_y = info.get("enemy_y_position", 0)

        # Calcular el fitness paso a paso
        fitness += fitness_funcion(
            (self_damage, self_x, self_y),
            (enemy_damage, enemy_x, enemy_y)
        )
        

        last_enemy_health = info['enemy_health']
        last_player_health = info['health']
        total_reward += reward
        frame_count += 1

        # Limitar a 60 FPS para que no vaya ultra rápido
        clock.tick(60)

    print(f"\n🏁 Partida terminada. Fitness total: {fitness}, Cuadros jugados: {frame_count}")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    jugar_humano()
