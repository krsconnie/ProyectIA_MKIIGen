import gymnasium as gym
import numpy as np
import retro
import cv2  # OpenCV para redimensionar imágenes
from gymnasium import spaces

class MortalKombatEnv(gym.Env):
    def __init__(self, resize_shape=(64, 64)):
        super().__init__()
        self.env = retro.make(game='MortalKombatII-Genesis', players=1, use_restricted_actions=retro.Actions.ALL)

        # Guardar tamaño deseado
        self.resize_shape = resize_shape

        # Redefinir espacio de observación (normalizado y redimensionado)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.resize_shape[1], self.resize_shape[0]),
            dtype=np.uint8
        )

        self.action_space = self.env.action_space

        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        self.distancia_anterior = 0

    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.uint8)
        # Cambiar a formato (C, H, W)
        return np.transpose(normalized, (2, 0, 1))


    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.preprocess(obs)

        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        self.distancia_anterior = 0

        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess(obs)

        p1_health_actual = info.get('health', self.p1_health_anterior)
        p2_health_actual = info.get('enemy_health', self.p2_health_anterior)

        reward = self.p2_health_anterior - p2_health_actual
        reward -= (self.p1_health_anterior - p1_health_actual)

        x = info.get('x_position', 0)
        x_enemy = info.get('enemy_x_position', 0)
        distancia_actual = abs(x - x_enemy)
        reward += (self.distancia_anterior - distancia_actual) * 0.01

        if info.get('rounds_won', 0) > 0:
            reward += 50
        if info.get('enemy_rounds_won', 0) > 0:
            reward -= 50

        self.p1_health_anterior = p1_health_actual
        self.p2_health_anterior = p2_health_actual
        self.distancia_anterior = distancia_actual

        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
