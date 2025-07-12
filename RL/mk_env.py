import gymnasium as gym
import numpy as np
import retro
import cv2
from gymnasium import spaces

class MortalKombatEnv(gym.Env):
    def __init__(self, resize_shape=(84, 84), n=4):
        super().__init__()

        # Creación de env en base a mapas aleatorios
        self.posibles_estados = [f"VeryEasy.LiuKang-{i:02d}" for i in range(2, 9)]
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
        self.no_atack_steps = 0
        self.prev_health = 120
        self.prev_enemy_hp = 120

        # Parámetros para frame skip
        self.n = n

        # Parámetros para estadísticas
        self.damage_to_player_steps = 0
        self.efective_attack_steps = 0
        self.efective_block_steps = 0
        self.total_steps = 0

        # Control para intro de batalla
        self.saltando_intro = False
        self.intro_steps_restantes = 175


    def _crear_nuevo_env(self):
        estado_random = np.random.choice(self.posibles_estados)
        self.env = retro.make(
            game='MortalKombatII-Genesis',
            state=estado_random,
            players=1,
            scenario='scenario'
            ,render_mode = False
        )


    # Método para procesar el cambio de tamaño de la imagen
    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))


    # Método de reset del entorno y las variables de recompensa
    def reset(self, seed=None, options=None):

        # Activar el salto de intro para los próximos step()
        self.saltando_intro = True
        self.intro_steps_restantes = 175

        # Cierra el entorno anterior si existe
        if self.env:
            self.env.close()
        self._crear_nuevo_env()

        obs, info = self.env.reset()
        obs = self.preprocess(obs)

        # Reset de las variables
        self.last_step_action = None
        self.last_step_info = info
        self.no_atack_steps = 0
        self.prev_health = 120
        self.prev_enemy_hp = 120

        # Reset de parámetros para estadísticas
        self.damage_to_player_steps = 0
        self.efective_attack_steps = 0
        self.efective_block_steps = 0
        self.total_steps = 0

        return obs, info


    def step(self, action):

        terminated = False
        truncated = False
        reward = 0

        # Saltar intro si está activo
        if self.saltando_intro:
            accion_neutral = [0] * self.action_space.shape[0]
            obs, _, terminated, truncated, info = self.env.step(accion_neutral)
            self.intro_steps_restantes -= 1

            if terminated or truncated or self.intro_steps_restantes <= 0:
                self.saltando_intro = False

            obs = self.preprocess(obs)
            return obs, 0.0, terminated, truncated, info

        # Frame skipping no estocástico
        for i in range(self.n):
            obs, _, terminated, truncated, info = self.env.step(action)
            obs = self.preprocess(obs)

            if terminated or truncated:
                break

        self.total_steps += 1

        # Obtención de variables actuales
        curr_health = info.get('health', self.prev_health)
        curr_enemy_hp = info.get('enemy_health', self.prev_enemy_hp)
        damage_to_Enemy = self.prev_enemy_hp - curr_enemy_hp
        damage_to_Player = self.prev_health - curr_health

        # attack buttons = B, A, C
        attack_buttons = [0, 1, 8]
        # block buttons = START
        block_buttons = [3]

        #Añadida nueva variable para atacar. Solo uno debe estar activo, sino no funcionan los ataques
        attack_pressed = [action[i] for i in attack_buttons]

        # Por cada step que genere daño, recompensa base de 0.5 mas daño generado
        if damage_to_Enemy > 0:
            reward += 0.6 + (damage_to_Enemy/120)
            self.efective_attack_steps += 1
        
        # Si el personaje no ataca en cierto tiempo, entonces será castigado. 
        # Se considerará solo si al personaje no se le hace daño, ya que puede estar inmovil debido a los golpes
        if curr_health == self.prev_health:
            if damage_to_Enemy == 0:
                self.no_atack_steps += 1
            else:
                self.no_atack_steps = 0

            if self.no_atack_steps >= 10:
                reward -= 0.06

        # Por cada step en que reciba daño, descuento de 0.2. Si bloquea, solo se le descuenta 0.15
        if damage_to_Player > 0:
            self.damage_to_player_steps += 1
            reward -= 0.2 + (damage_to_Player/120)

            if any(action[i] for i in block_buttons) and sum(attack_pressed) == 0:
                reward += 0.05
                self.efective_block_steps += 1

        # Actualización de varibles anteriores
        self.prev_health = curr_health
        self.prev_enemy_hp = curr_enemy_hp
        self.last_step_action = action
        self.last_step_info = info
        
        if terminated or truncated:
                #Personaje perdió
                if curr_health <= 0:
                    reward -= 5
                #Personaje ganó
                if curr_enemy_hp <= 0:
                    reward += 10


        info["efective_attack_steps"] = self.efective_attack_steps
        info["efective_block_steps"] = self.efective_block_steps
        info["total_steps"] = self.total_steps
        info["damage_to_player_steps"] = self.damage_to_player_steps

        done = terminated or truncated
        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
