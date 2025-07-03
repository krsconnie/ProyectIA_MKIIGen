from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None
        self.episode_rewards = []

    def _on_training_start(self):
        self.writer = SummaryWriter(log_dir=self.logger.dir)

    def _on_step(self):
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        infos = self.locals['infos']

        for i in range(len(rewards)):
            if len(self.episode_rewards) <= i:
                self.episode_rewards.append(0.0)

            self.episode_rewards[i] += rewards[i]

            if dones[i]:
                # 🧠 Obtener estadísticas directamente del info[i]
                info = infos[i]
                efective_attack_steps = info.get("efective_attack_steps", 0)
                efective_block_steps = info.get("efective_block_steps", 0)
                total_steps = max(info.get("total_steps", 1), 1)
                steps_cerca = info.get("steps_cerca_del_enemigo", 0)
                damage_to_player = info.get("damage_to_player_steps", 0)

                # 🔢 Porcentajes
                pct_attack = efective_attack_steps / total_steps
                pct_block = efective_block_steps / total_steps
                pct_cerca = steps_cerca / total_steps
                pct_dano = damage_to_player / total_steps

                # 📊 Logging en TensorBoard: Porcentajes
                self.writer.add_scalar("episode/reward", self.episode_rewards[i], self.num_timesteps)
                self.writer.add_scalar("episode/pct_attack", pct_attack, self.num_timesteps)
                self.writer.add_scalar("episode/pct_block", pct_block, self.num_timesteps)
                self.writer.add_scalar("episode/pct_cerca", pct_cerca, self.num_timesteps)
                self.writer.add_scalar("episode/pct_dano", pct_dano, self.num_timesteps)

                # 📊 Logging en TensorBoard: Valores absolutos
                self.writer.add_scalar("raw/attack_steps", efective_attack_steps, self.num_timesteps)
                self.writer.add_scalar("raw/block_steps", efective_block_steps, self.num_timesteps)
                self.writer.add_scalar("raw/damage_to_player", damage_to_player, self.num_timesteps)
                self.writer.add_scalar("raw/steps_cerca", steps_cerca, self.num_timesteps)
                self.writer.add_scalar("raw/total_steps", total_steps, self.num_timesteps)

                # Reset de reward
                self.episode_rewards[i] = 0.0

        return True

    def _on_training_end(self):
        self.writer.close()
