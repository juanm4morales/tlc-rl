import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from tscRL.environments.environment import SumoEnvironment
from .callbacks import CustomMetricsCallback

from typing import Tuple
class DQNAgent:
    def __init__(
        self,
        env: SumoEnvironment,
        learningRate: float,
        bufferSize: int = 1_000_000, 
        batchSize: int = 64,
        gamma: float = 0.99,
        explorationFraction: float = 0.5,
        targetUpdateInterval: int = 1000,
        initialEpsilon: float = 1,
        finalEpsilon: float = 0.01,
        netArch: Tuple[int, int] = (32,32),
        verbose: int = 0,
        callback: BaseCallback = CustomMetricsCallback(verbose=1)
    ) -> None:
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learningRate,
            buffer_size=bufferSize,
            batch_size=batchSize,
            gamma=gamma,
            train_freq=1,
            target_update_interval=targetUpdateInterval,
            exploration_fraction=explorationFraction,
            exploration_initial_eps=initialEpsilon,
            exploration_final_eps=finalEpsilon,
            policy_kwargs=dict(net_arch=[netArch[0], netArch[1]]),
            verbose=verbose
        )
        self.steps_per_episode = env.totalTimeSteps
        self.callback = callback
    
    def learn(self, episodes: int = 50, logInterval: int = 1, progressBar: bool = False):
        
        total_timesteps = episodes * self.steps_per_episode
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            log_interval=logInterval,
            reset_num_timesteps=True,
            progress_bar=progressBar
        )
        
    
    def setModel(self, env: SumoEnvironment):
        if self.model.get_env().observation_space != env.observation_space:
            self.model = DQN(
                policy="MlpPolicy",
                env=env,
                learning_rate=self.model.learning_rate,
                buffer_size=self.model.buffer_size,
                batch_size=self.model.batch_size,
                gamma=self.model.gamma,
                train_freq=1,
                target_update_interval=1000,
                exploration_fraction=self.model.exploration_fraction,
                exploration_initial_eps=self.model.exploration_initial_eps,
                exploration_final_eps=self.model.exploration_final_eps,
                policy_kwargs=self.model.policy_kwargs,
                verbose=self.model.verbose
            )
  
        else:
            self.model.set_env(env)
        
        