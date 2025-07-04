import retro
import os
import gymnasium as gym
import numpy as np
import csv
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


# Configuraciones

USE_PARALLEL = False  # True para activar entrenamiento paralelo
NUM_ENVS = 4  # Número de entornos 
TOTAL_TIMESTEPS = 50000
MODEL_PATH = "mk2_agent"
LOG_CSV = "stats.csv"


class MortalKombatEnv(gym.Env):
    def __init__( self, n = 4 ):
        super().__init__()
        self.env = retro.make( game = 'MortalKombatII-Genesis', state = 'VeryEasy.LiuKang-02', players = 1, scenario = 'scenario')
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.prev_enemy_health = None
        self.prev_health = None

        self.failed_attacks = 0
        self.n = n
        self.episode_rewards = 0.0
        self.successful_blocks = 0
        self.successful_hits = 0
        self.start_time = None

    def reset( self, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        for _ in range(175):
            obs, _, terminated, truncated, info = self.env.step(self.env.action_space.sample())
           
            if terminated or truncated:
                obs, info = self.env.reset()
        
        self.prev_enemy_health = self.env.data.lookup_value('enemy_health')
        self.prev_health = self.env.data.lookup_value('health')
        self.failed_attacks = 0
        self.episode_rewards = 0.0
        self.successful_blocks = 0
        self.successful_hits = 0
        self.start_time = datetime.now()
        
        return obs, {}

    def step( self, action ):
        
        terminated = False
        truncated = False
        totrew = 0

        for i in range(self.n):
        
            obs, _, terminated, truncated, info = self.env.step(action)
        
            if i == 0:
        
                current_enemy_hp = info.get("enemy_health", 0)
                current_health = info.get("health", 0)
                reward = 0.0

                attack_buttons = [0, 1, 8]
                block_buttons = [3]

                delta_enemy = self.prev_enemy_health - current_enemy_hp
                delta_self = self.prev_health - current_health

                if delta_enemy > 0:
        
                    reward += 0.6
                    self.failed_attacks = 0
                    self.successful_hits += 1
        
                else:
        
                    if any(action[i] for i in attack_buttons):
        
                        self.failed_attacks += 1
                        reward -= 0.5

                if self.failed_attacks >= 3:
        
                    reward -= 0.5
                    self.failed_attacks = 0

                if delta_self > 0:
        
                    reward -= 0.8
        
                    if any(action[i] for i in block_buttons) and delta_self < 5:
        
                        reward += 0.6
                        self.successful_blocks += 1

                self.prev_enemy_health = current_enemy_hp
                self.prev_health = current_health
        
            else:
        
                reward = 0.0

            totrew += reward
        
            if terminated or truncated:
                break

        done = terminated or truncated

        if done:
        
            final_enemy_hp = info.get("enemy_health", 0)
            final_self_hp = info.get("health", 0)
        
            if final_enemy_hp <= 0 and final_self_hp > 0:
                totrew += 10.0
        
            elif final_self_hp <= 0:
                totrew -= 5.0

            self.episode_rewards += totrew
            self.log_episode(final_enemy_hp, final_self_hp)

        self.episode_rewards += totrew
        
        return obs, totrew, done, False, info

    def log_episode(self, enemy_hp, player_hp):
        
        duration = (datetime.now() - self.start_time).total_seconds()
        with open(LOG_CSV, "a", newline='') as f:
        
            writer = csv.writer(f)
            writer.writerow([
        
                datetime.now().isoformat(),
                round(self.episode_rewards, 2),
                self.successful_hits,
                self.successful_blocks,
                duration,
                player_hp,
                enemy_hp
            ])

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


# Crea entorno 

def make_env(rank):
    
    def _init():
        return MortalKombatEnv()
    
    return _init

if USE_PARALLEL:
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

else:
    env = DummyVecEnv([make_env(0)])

env = VecMonitor(env)


if os.path.exists(MODEL_PATH + ".zip"):
    print("Carlos 6to aún vive...")
    model = PPO.load(MODEL_PATH, env=env)

else:
    print("Entrenando modelo desde cero, creabdo un nuevo carlos...")
    model = PPO("CnnPolicy", env, verbose=1)

model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_PATH)
print(f"Modelo guardado como '{MODEL_PATH}.zip'")