import os
import csv
import retro
import numpy as np
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from torch.utils.tensorboard import SummaryWriter
import cv2

# === CONFIGURACIÓN DE ENTRENAMIENTO ===
NUM_ENVS = 2 # Reducido para menos uso de RAM/CPU
TOTAL_TIMESTEPS = 10_000  # Reducido para pruebas iniciales
MODEL_PATH = "mk2_agent"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_CSV = f"stats_{timestamp}.csv"
TENSORBOARD_LOG_DIR = f"./RL/tensorboard_logs/{timestamp}"
CHECKPOINT_DIR = "./RL/checkpoints"

# === ENTORNO PERSONALIZADO OPTIMIZADO ===
class MortalKombatEnv(gym.Env):
    def __init__(self, n=4, resize_shape=(84, 84)):
        super().__init__()
        self.env = retro.make(
            game='MortalKombatII-Genesis',
            state='Level1.JaxVsLiuKang',
            scenario='scenario'
            #render_mode='rgb_array'
        )
        self.action_space = self.env.action_space
        self.resize_shape = resize_shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, resize_shape[1], resize_shape[0]), dtype=np.uint8
        )
        self.n = n
        self.reset_stats()

    def reset_stats(self):
        self.prev_enemy_health = 120
        self.prev_health = 120
        self.failed_attacks = 0
        self.episode_rewards = 0.0
        self.successful_blocks = 0
        self.successful_hits = 0
        self.consecutive_hits = 0  # Nuevo: contador de golpes consecutivos
        self.start_time = datetime.now()
        self.block_spam_counter = 0  # Nuevo: contar bloqueos seguidos inútiles
        self.last_block_success = False

    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        for _ in range(175):
            obs, _, terminated, truncated, info = self.env.step([0]*self.action_space.shape[0])
            if terminated or truncated:
                obs, info = self.env.reset()
        self.reset_stats()
        return self.preprocess(obs), {}

    def step(self, action):
        terminated = False
        truncated = False
        total_reward = 0

        for i in range(self.n):
            obs, _, terminated, truncated, info = self.env.step(action)
            if i == 0:
                reward = self.compute_reward(action, info)
            else:
                reward = 0
            total_reward += reward
            if terminated or truncated:
                break

        self.episode_rewards += total_reward
        obs = self.preprocess(obs)
        done = terminated or truncated

        if done:
            self.log_episode(info)

        return obs, total_reward, done, False, info

    def compute_reward(self, action, info):
        current_enemy_hp = info.get("enemy_health", 0)
        current_health = info.get("health", 0)
        reward = 0.0

        delta_enemy = self.prev_enemy_health - current_enemy_hp
        delta_self = self.prev_health - current_health

        attack_buttons = [0, 1, 8]
        block_buttons = [3]
        jump_buttons = [4]

        # ===== ATAQUE =====
        if delta_enemy > 0:
            self.failed_attacks = 0
            self.successful_hits += 1
            self.consecutive_hits += 1

            base_reward = 1.5 * min(delta_enemy / 30, 1.0)
            combo_bonus = 0.1 * (2 ** (self.consecutive_hits - 1) - 1)
            reward += base_reward + combo_bonus
        else:
            if any(action[i] for i in attack_buttons):
                self.failed_attacks += 1
                reward -= 0.05
            self.consecutive_hits = 0

        if self.failed_attacks >= 4:
            reward -= 0.025
            self.failed_attacks = 0

        # ===== BLOQUEO =====
        blocking = any(action[i] for i in block_buttons)
        if delta_self > 0:  # recibió daño
            reward -= 1.0 * min(delta_self / 30, 1.0)
            if blocking and delta_self < 5:  # bloqueó parcialmente
                reward += 0.05
                self.successful_blocks += 1
                self.block_spam_counter = 0
                self.last_block_success = True
            else:
                self.last_block_success = False
        else:
            if blocking:
                reward += 0.1  # aún útil bloquear de vez en cuando
                if not self.last_block_success:
                    self.block_spam_counter += 1
                    if self.block_spam_counter > 5:
                        reward -= 0.3  # penalizar spam de bloqueo
                else:
                    self.block_spam_counter = 0
            else:
                self.block_spam_counter = 0
            self.last_block_success = False

        # ===== SALTOS =====
        if any(action[i] for i in jump_buttons):
            reward -= 0.05

        self.prev_enemy_health = current_enemy_hp
        self.prev_health = current_health

        return reward


    def log_episode(self, info):
        duration = (datetime.now() - self.start_time).total_seconds()
        final_enemy_hp = info.get("enemy_health", 0)
        final_self_hp = info.get("health", 0)

        with open(LOG_CSV, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                round(self.episode_rewards, 2),
                self.successful_hits,
                self.successful_blocks,
                duration,
                final_self_hp,
                final_enemy_hp
            ])

    def render(self):
        # No render para ahorrar recursos
        pass

    def close(self):
        self.env.close()


# === CALLBACK PARA TENSORBOARD ===
class RewardTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            if "health" in info:
                self.writer.add_scalar("Custom/Health", info["health"], self.num_timesteps)
            if "enemy_health" in info:
                self.writer.add_scalar("Custom/EnemyHealth", info["enemy_health"], self.num_timesteps)
        return True

    def _on_training_end(self):
        self.writer.close()


def make_env(rank):
    def _init():
        return MortalKombatEnv()
    return _init


if __name__ == "__main__":
    import torch

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TENSORBOARD_LOG_DIR), exist_ok=True)

    with open(LOG_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "episode_reward", "successful_hits", "successful_blocks",
            "duration", "player_hp", "enemy_hp"
        ])

    print("✅ GPU disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("💻 Usando:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ Usando CPU. Revisa instalación de CUDA si esperabas usar GPU.")

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecMonitor(env)

    if os.path.exists(MODEL_PATH + ".zip"):
        print("📦 Cargando modelo existente...")
        model = PPO.load(MODEL_PATH, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("🧠 Creando nuevo modelo PPO...")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_mk2"
    )
    reward_callback = RewardTensorboardCallback()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([checkpoint_callback, reward_callback]),
        reset_num_timesteps=False
    )

    model.save(MODEL_PATH)
    print(f"✅ Modelo guardado como '{MODEL_PATH}.zip'")