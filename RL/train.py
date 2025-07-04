import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from mk_env import MortalKombatEnv
from stable_baselines3.common.callbacks import CallbackList
from reward_logger_callback import RewardLoggerCallback
from datetime import datetime

def main():
    num_procesos = min(multiprocessing.cpu_count(), 4)
    print(f"Usando {num_procesos} procesos para entrenar...")

    # Crear carpetas si no existen
    checkpoint_dir = "./RL/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SubprocVecEnv([lambda: MortalKombatEnv() for _ in range(num_procesos)])

    model_path = "./RL/ppo_mk_model"
    log_name = datetime.now().strftime("PPO_%d-%m___%H-%M_VeryEasy-02")

    if os.path.exists(f"{model_path}.zip"):
        print("Cargando modelo base existente y continuando entrenamiento...")
        model = PPO.load(model_path, env=env, tensorboard_log="./RL/tensorboard_logs")
    else:
        print("Creando modelo nuevo...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=2.5e-4,
            ent_coef=0.015,
            clip_range=0.1,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.4,
            verbose=1,
            tensorboard_log="./RL/tensorboard_logs"
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_mk_checkpoint"
    )
    reward_logger_callback = RewardLoggerCallback()
    callback_list = CallbackList([checkpoint_callback, reward_logger_callback])

    print("Comenzando entrenamiento...")
    model.learn(
        total_timesteps= 1_000_000 * num_procesos,
        callback=callback_list,
        reset_num_timesteps=False,
        tb_log_name=log_name
    )

    # Guardar modelo final
    model.save(model_path)
    print(f"-----------Entrenamiento finalizado y modelo guardado como {model_path}.zip------------")

if __name__ == '__main__':
    main()
