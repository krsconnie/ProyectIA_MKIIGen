import gymnasium as gym
import numpy as np
import retro
import cv2  # OpenCV para redimensionar imágenes
from gymnasium import spaces

class MortalKombatEnv(gym.Env):
    def __init__(self, resize_shape=(64, 64)):
        super().__init__()
        self.env = retro.make(game='MortalKombatII-Genesis', players=1)
        self.resize_shape = resize_shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.resize_shape[1], self.resize_shape[0]),
            dtype=np.uint8
        )
        self.action_space = self.env.action_space

        self.last_step_info = None

        self.reward = 0
        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        self.distancia_anterior = 103  # Distancia de inicio

    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))

    def reset(self, seed=None, options=None):
        print("RESEEEEEEEEET")
        obs, info = self.env.reset(seed=seed, options=options)

        for _ in range(175):
            obs, _, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)

        obs = self.preprocess(obs)

        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        x = info.get('x_position', 0)
        x_enemy = info.get('enemy_x_position', 0)
        self.distancia_anterior = abs(x - x_enemy) - 45
        self.reward = 0

        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess(obs)

        p1_health_actual = info.get('health', self.p1_health_anterior)
        p2_health_actual = info.get('enemy_health', self.p2_health_anterior)

        damage_to_Enemy = self.p2_health_anterior - p2_health_actual
        damage_to_Player = self.p1_health_anterior - p1_health_actual

        #Por cada step que genere daño, recompensa del daño hecho + 10 (minimo daño posible es 5, maximo no se sabe, pero asumimos que 30)
        if damage_to_Enemy > 0:
            print("ATAQUE NUESTRO >:D")
            self.reward += (damage_to_Enemy + 10)*3

        # Por cada step en que reciba daño, descuento del daño realizado
        if damage_to_Player > 0:
            print("ATAQUE ENEMIGO D:")
            self.reward -= damage_to_Player

        x = info.get('x_position', 0)
        x_enemy = info.get('enemy_x_position', 0)
        distancia_actual = abs(x - x_enemy) - 45 #La distancia min entre jugadores es de 45
        
        print(f"Diferencia de distancias entre steps: {self.distancia_anterior - distancia_actual}")
        self.reward += (self.distancia_anterior - distancia_actual) * 0.04 #El cambio de distancias max es desde 199 a 0, lo cual da recompensa max de 7.96

        if damage_to_Enemy == 0 and damage_to_Player == 0 and distancia_actual == self.distancia_anterior:
            self.reward -= 0.005  # castigo leve por no hacer nada

        if terminated or truncated:
            print(f"p1 = {p1_health_actual}, p2 ={p2_health_actual}")
            #Personaje perdió
            if p1_health_actual <= 0:
                self.reward -= 100
            if p2_health_actual <= 0:
                self.reward += 500

        self.p1_health_anterior = p1_health_actual
        self.p2_health_anterior = p2_health_actual
        self.distancia_anterior = distancia_actual
        print(f"REWARD: {self.reward}")

        return obs, self.reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
