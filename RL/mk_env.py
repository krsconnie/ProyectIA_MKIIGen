import gymnasium as gym
import numpy as np
import retro
import cv2  # OpenCV para redimensionar imágenes
from gymnasium import spaces

class MortalKombatEnv(gym.Env):
    def __init__(self, resize_shape=(84, 84), n=4):
        super().__init__()
        self.posibles_estados = [f"VeryEasy.LiuKang-{i:02d}" for i in range(2, 4)]
        self.env = None
        self._crear_nuevo_env()
        
        
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
        self.combo_golpes_consecutivos = 0
        self.steps_estatico = 0
        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        self.distancia_anterior = 103  # Distancia de inicio

        # Parámetros para stochastic frame skip
        self.n = n

        # Parámetros para estadísticas
        self.efective_attack_steps = 0
        self.efective_block_steps = 0
        self.total_steps = 0
        self.steps_cerca_del_enemigo = 0
        self.damage_to_player_steps = 0

    def _crear_nuevo_env(self):
        estado_random = np.random.choice(self.posibles_estados)
        self.env = retro.make(
            game='MortalKombatII-Genesis',
            state=estado_random,
            players=1,
            scenario='scenario'
            #,render_mode = False
        )

    # Método para procesar el cambio de tamaño de la imagen
    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))

    # Método de reset del entorno y las variables de recompensa
    def reset(self, seed=None, options=None):
        # Cierra el entorno anterior si existe
        if self.env:
            self.env.close()

        # Crear nuevo estado aleatorio
        self._crear_nuevo_env()

        obs, info = self.env.reset()

        # Tiempo de espera para la intro de batalla del juego
        for _ in range(175):
            obs, _, terminated, truncated, info = self.env.step(self.action_space.sample())
            if terminated or truncated:
                obs, info = self.env.reset()

        obs = self.preprocess(obs)

        # Reset de las variables
        self.steps_estatico = 0
        self.last_step_action = None
        self.last_step_info = info
        self.acciones_repetidas = 0
        self.combo_golpes_consecutivos = 0
        self.p1_health_anterior = 120
        self.p2_health_anterior = 120
        x = info.get('x_position', 0)
        x_enemy = info.get('enemy_x_position', 0)
        self.distancia_anterior = abs(x - x_enemy) - 45  #La distancia min entre jugadores es de 45

        # Reset de parámetros para estadísticas
        self.efective_attack_steps = 0
        self.efective_block_steps = 0
        self.total_steps = 0
        self.steps_cerca_del_enemigo = 0
        self.damage_to_player_steps = 0

        return obs, info


    def step(self, action):

        #action = action[:12]

        terminated = False
        truncated = False
        reward = 0

        # Frame skipping no estocástico
        for i in range(self.n):
            obs, _, terminated, truncated, info = self.env.step(action)
            obs = self.preprocess(obs)

            if terminated or truncated:
                break

        self.total_steps += 1

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

        #Añadida nueva variable para atacar. Solo uno debe estar activo, sino no funcionan los ataques
        attack_pressed = [action[i] for i in attack_buttons]

        # Por cada step que genere daño, recompensa base de 0.2 más combo bonus
        if damage_to_Enemy > 0:
            self.combo_golpes_consecutivos += 1
            bonus_combo = min(0.05 * self.combo_golpes_consecutivos, 0.15)
            reward += 0.45 + bonus_combo + (damage_to_Enemy/120)
            self.efective_attack_steps += 1
        else:
            self.combo_golpes_consecutivos = 0

        # Por cada step en que reciba daño, descuento de 0.3. Si bloquea, solo se le descuenta 0.2
        if damage_to_Player > 0:
            self.damage_to_player_steps += 1
            reward -= 0.06 + (damage_to_Player/120)
            if any(action[i] for i in block_buttons) and sum(attack_pressed) == 0:
                #Tentativo a cambia a 0.01 si la IA se vuelve muy defensiva
                reward += 0.008
                self.efective_block_steps += 1

        # Si golpea, se le recompensa un poco aunque no genere daño, pero debe estar a una distancia de 15
        if sum(attack_pressed) == 1 and distancia_actual <= 15 and p1_health_actual == self.p1_health_anterior:
            reward += 0.6
        
        # Si el jugador se acerca al oponente, se le recompensa
        if distancia_actual <= self.distancia_anterior:
            if (x < x_enemy and action[7] == 1) or (x > x_enemy and action[6] == 1):
                reward += (self.distancia_anterior - distancia_actual) * 0.002 #El cambio de distancias max es desde 199 a 0, lo cual da recompensa max de 0.199
        
        # Si el jugador se aleja del oponente, sin intensión de apartarse de golpes, se le castiga 
        else:
            if self.p1_health_anterior == p1_health_actual and ((x < x_enemy and action[6] == 1) or (x > x_enemy and action[7] == 1)):
                reward -= (distancia_actual - self.distancia_anterior) * 0.0018 #El cambio de distancias max es desde 199 a 0, lo cual da recompensa max de 0.2985

        # Si el personaje no produce ningún efecto, entonces será castigado tras una serie de pasos. 
        # Se considerará solo si al personaje no se le hace daño, ya que puede estar inmovil debido a los golpes
        if p1_health_actual == self.p1_health_anterior:
            if damage_to_Enemy == 0:
                self.acciones_repetidas += 1
            else:
                self.acciones_repetidas = 0

            if self.acciones_repetidas >= 30:
                reward -= min(0.05 * (self.acciones_repetidas - 29), 0.5)

        # Si el personaje se queda en el mismo sitio por mucho tiempo, sin que le hagan daño, 
        # tendrá que moverse o será castigado. Con esto se busca corregir el comportamiento de quedarse pegado al fondo del mapa
        # Saltando y golpeando al enemigo (Forma fácil de ganarle, ya que las otras IAs no son tan inteligentes para defenderse ahí)
        if p1_health_actual == self.p1_health_anterior:
            if self.last_step_info.get('x_position', 0) == x:
                self.steps_estatico += 1
            else:
                self.steps_estatico = 0
            if self.steps_estatico >= 40:
                #Cambio de 1 a 0.06
                reward -= min(0.02 * (self.steps_estatico - 39), 0.04)
        
        if distancia_actual < 5:
            self.steps_cerca_del_enemigo += 1

        # Actualización de varibles anteriores
        self.p1_health_anterior = p1_health_actual
        self.p2_health_anterior = p2_health_actual
        self.distancia_anterior = distancia_actual
        self.last_step_action = action
        self.last_step_info = info
        
        if terminated or truncated:
                #Personaje perdió
                if p1_health_actual <= 0:
                    reward -= 1
                #Personaje ganó
                if p2_health_actual <= 0:
                    reward += 1


        info["efective_attack_steps"] = self.efective_attack_steps
        info["efective_block_steps"] = self.efective_block_steps
        info["total_steps"] = self.total_steps
        info["steps_cerca_del_enemigo"] = self.steps_cerca_del_enemigo
        info["damage_to_player_steps"] = self.damage_to_player_steps


        done = terminated or truncated
        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
