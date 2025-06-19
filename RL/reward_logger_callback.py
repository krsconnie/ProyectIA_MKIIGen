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
        # Sumamos la recompensa
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        for i in range(len(dones)):
            if len(self.episode_rewards) <= i:
                self.episode_rewards.append(0.0)
            self.episode_rewards[i] += rewards[i]
            if dones[i]:
                self.writer.add_scalar("episode/reward", self.episode_rewards[i], self.num_timesteps)
                self.episode_rewards[i] = 0.0
        return True

    def _on_training_end(self):
        self.writer.close()
