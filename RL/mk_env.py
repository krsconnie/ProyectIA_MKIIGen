import gymnasium as gym
import numpy as np
import retro
import cv2  # OpenCV para redimensionar imágenes
from gymnasium import spaces

class MortalKombatEnv(gym.Env):
    def __init__(self, resize_shape=(64, 64), n=4, stickprob=0.25):
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
        self.last_step_curac = None
        self.last_step_info = None
        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        self.distancia_anterior = 103  # Distancia de inicio

        # Parámetros para stochastic frame skip
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()

    # Método para procesar el cambio de tamaño de la imagen
    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))

    # Método de reset del entorno y las variables de recompensa
    def reset(self, seed=None, options=None):
        print("RESEEEEEEEEET")
        obs, info = self.env.reset()
        self.last_step_action = 0
        self.last_step_info = info

        # Tiempo de espera para la intro de batalla del juego
        for _ in range(175):
            obs, _, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            if terminated or truncated:
                obs, info = self.env.reset()

        obs = self.preprocess(obs)

        # Reset de las variables
        self.curac = None
        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        x = info.get('x_position', 0)
        x_enemy = info.get('enemy_x_position', 0)
        self.distancia_anterior = abs(x - x_enemy) - 45  #La distancia min entre jugadores es de 45

        return obs, info


    def step(self, action):

        terminated = False
        truncated = False
        total_Reward = 0

        for i in range(self.n):
            if self.curac is None:
                self.curac = action
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = action
            elif i == 1:
                self.curac = action

            obs, _, terminated, truncated, info = self.env.step(self.curac)
            obs = self.preprocess(obs)

            # Obtención de variables actuales
            p1_health_actual = info.get('health', self.p1_health_anterior)
            p2_health_actual = info.get('enemy_health', self.p2_health_anterior)
            damage_to_Enemy = self.p2_health_anterior - p2_health_actual
            damage_to_Player = self.p1_health_anterior - p1_health_actual
            x = info.get('x_position', 0)
            x_enemy = info.get('enemy_x_position', 0)
            distancia_actual = abs(x - x_enemy) - 45 

            reward = 0
            # Recompensa calculada solo para el paso principal
            if i == 0:
                # attack buttons = B, A, C
                attack_buttons = [0, 1, 8]
                # block buttons = START
                block_buttons = [3]
                # Movement_buttons = UP, DOWN, LEFT, RIGHT
                movement_buttons = [4, 5, 6, 7]

                # Por cada step que genere daño, recompensa del daño hecho + 10 
                # (minimo daño posible es 5, maximo asumimos que 30) por 3
                if damage_to_Enemy > 0:
                    print("ATAQUE NUESTRO :D")
                    reward += ((damage_to_Enemy + 10)*3) #Rango de 45 a 120        #/ 100 # Rango de 0.45 a 1.2
                
                """
                # En caso de no generar daño, pero haber atacado, ligero castigo de - 5
                else:
                    if any(action[i] for i in attack_buttons):
                        reward -= 5
                """

                # Por cada step en que reciba daño, descuento de 20. Si bloquea, solo se le descuenta 15
                if damage_to_Player > 0:
                    print("ATAQUE ENEMIGO D:")
                    reward -= damage_to_Player
                    if any(action[i] for i in block_buttons):
                        reward += 5
                
                # Si el jugador se acerca al oponente, se le recompensa
                if distancia_actual <= self.distancia_anterior:
                    if x < x_enemy and action[7] == 1:
                        reward += (self.distancia_anterior - distancia_actual) * 0.04 #El cambio de distancias max es desde 199 a 0, lo cual da recompensa max de 7.96
                    if x > x_enemy and action[6] == 1:
                        reward += (self.distancia_anterior - distancia_actual) * 0.04 
                # Si el jugador se aleja del oponente, se le castiga 
                else:
                    if x < x_enemy and action[6] == 1:
                        reward -= (self.distancia_anterior - distancia_actual) * 0.04
                    if x > x_enemy and action[7] == 1:
                        reward -= (self.distancia_anterior - distancia_actual) * 0.04 

                # Si el jugador no se mueve será castigado
                #if self.last_step_info.get('x_position',0) == x:
                #    reward -= 0.05 

                # Si el jugador solo se queda agachado sin hacer ni recibir daño, se le castiga
                if damage_to_Enemy == 0 and damage_to_Player == 0 and distancia_actual == self.distancia_anterior and action[5] == 1:
                    reward -= 10  
                
            else:
                reward = 0
            
            # Actualización de varibles anteriores
            self.p1_health_anterior = p1_health_actual
            self.p2_health_anterior = p2_health_actual
            self.distancia_anterior = distancia_actual
            self.last_step_info = info
            self.last_step_curac = self.curac
            total_Reward += reward
            
            if terminated or truncated:
                break  # salimos del bucle si terminó el episodio
        
        if terminated or truncated:
                print(f"p1 = {p1_health_actual}, p2 ={p2_health_actual}")
                #Personaje perdió
                if p1_health_actual <= 0:
                    total_Reward -= 200
                if p2_health_actual <= 0:
                    total_Reward += 500
        
        print(f"REWARD: {total_Reward}")

        done = terminated or truncated
        return obs, total_Reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
