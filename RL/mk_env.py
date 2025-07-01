import gymnasium as gym
import numpy as np
import retro
import cv2  # OpenCV para redimensionar imágenes
from gymnasium import spaces

class MortalKombatEnv(gym.Env):
    def __init__(self, resize_shape=(64, 64), n=4):
        super().__init__()
        self.env = retro.make(game='MortalKombatII-Genesis', players=1, scenario='scenario')
        
        # Inicialización del entorno y del procesamiento
        self.resize_shape = resize_shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.resize_shape[1], self.resize_shape[0]),
            dtype=np.uint8
        )
        self.action_space = self.env.action_space

        # Variables para la evaluación de recompensas
        self.last_step_action = None
        self.last_step_info = None
        self.acciones_repetidas = 0
        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        self.distancia_anterior = 103  # Distancia de inicio

        # Parámetros para stochastic frame skip
        self.n = n

    # Método para procesar el cambio de tamaño de la imagen
    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))

    # Método de reset del entorno y las variables de recompensa
    def reset(self, seed=None, options=None):
        print("RESEEEEEEEEET")
        obs, info = self.env.reset()
        
        # Tiempo de espera para la intro de batalla del juego
        for _ in range(175):
            obs, _, terminated, truncated, info = self.env.step(self.action_space.sample())
            if terminated or truncated:
                obs, info = self.env.reset()

        obs = self.preprocess(obs)

        # Reset de las variables
        self.last_step_action = None
        self.last_step_info = info
        self.acciones_repetidas = 0
        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        x = info.get('x_position', 0)
        x_enemy = info.get('enemy_x_position', 0)
        self.distancia_anterior = abs(x - x_enemy) - 45  #La distancia min entre jugadores es de 45

        return obs, info


    def step(self, action):

        terminated = False
        truncated = False
        reward = 0

        # Frame skipping no estocástico
        for i in range(self.n):
            obs, _, terminated, truncated, info = self.env.step(action)
            obs = self.preprocess(obs)

            if terminated or truncated:
                break

        # Obtención de variables actuales
        p1_health_actual = info.get('health', self.p1_health_anterior)
        p2_health_actual = info.get('enemy_health', self.p2_health_anterior)
        damage_to_Enemy = self.p2_health_anterior - p2_health_actual
        damage_to_Player = self.p1_health_anterior - p1_health_actual
        x = info.get('x_position', 0)
        x_enemy = info.get('enemy_x_position', 0)
        distancia_actual = abs(x - x_enemy) - 45 

        # attack buttons = B, A, C
        attack_buttons = [0, 1, 8]
        # block buttons = START
        block_buttons = [3]
        # Movement_buttons = UP, DOWN, LEFT, RIGHT
        movement_buttons = [4, 5, 6, 7]

        # Por cada step que genere daño, recompensa del daño hecho + 10 
        # (minimo daño posible es 5, maximo asumimos que 30) por 3
        if damage_to_Enemy > 0:
            #print(f"ATAQUE NUESTRO :D {((damage_to_Enemy + 10)*3)}")
            reward += ((damage_to_Enemy + 10)*3) #Rango de 45 a 120        #/ 100 # Rango de 0.45 a 1.2

        # Por cada step en que reciba daño, descuento de 20. Si bloquea, solo se le descuenta 15
        if damage_to_Player > 0:
            #print(f"ATAQUE ENEMIGO D: {-(damage_to_Player + 10)}")
            reward -= (damage_to_Player + 10)
            if any(action[i] for i in block_buttons):
                #print("BLOQUEO, +5")
                reward += 5

        # Si golpea, se le recompensa un poco aunque no genere daño, pero debe estar a una distancia de 45
        if any(action[i] for i in (attack_buttons)) and distancia_actual <= 10:
            print("RRECOMPENSA POR ATACAR EN CUALQUIER MOMENTO")
            reward += 1
        
        # Si el jugador se acerca al oponente, se le recompensa
        if distancia_actual <= self.distancia_anterior:
            if x < x_enemy and action[7] == 1 or x > x_enemy and action[6] == 1:
                reward += (self.distancia_anterior - distancia_actual) * 0.02 #El cambio de distancias max es desde 199 a 0, lo cual da recompensa max de 7.96
                #print(f"RECOMPENSA POR ACERCARSE {(self.distancia_anterior - distancia_actual) * 0.02}")
        
        # Si el jugador se aleja del oponente, sin intensión de apartarse de golpes, se le castiga 
        else:
            if self.p1_health_anterior == p1_health_actual:
                reward -= (distancia_actual - self.distancia_anterior) * 0.025
                #print(f"DESRECOMPENSA POR ALEJARSE {(distancia_actual - self.distancia_anterior) * -0.025}")

        # Si el personaje solo repite las mismas acciones y no produce ningún efecto, entonces será castigado tras una serie de ellos
        if any(action[i] for i in (movement_buttons)) and not any(action[i] for i in attack_buttons) and damage_to_Enemy == 0:
            self.acciones_repetidas += 1
        else:
            self.acciones_repetidas = 0

        if self.acciones_repetidas >= 5:
            reward -= 5 * (self.acciones_repetidas - 4)
            print("DESRECOMPENSA POR PASIVO")
        
        # Actualización de varibles anteriores
        self.p1_health_anterior = p1_health_actual
        self.p2_health_anterior = p2_health_actual
        self.distancia_anterior = distancia_actual
        self.last_step_action = action
        self.last_step_info = info
        
        if terminated or truncated:
                print(f"p1 = {p1_health_actual}, p2 ={p2_health_actual}")
                #Personaje perdió
                if p1_health_actual <= 0:
                    reward -= 100
                if p2_health_actual <= 0:
                    reward += 200
    
        print(f"REWARD: {reward}")

        done = terminated or truncated
        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
