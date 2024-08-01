import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

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
        callback: BaseCallback = None,
    ) -> None:
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learningRate,
            buffer_size=bufferSize,
            batch_size=batchSize,
            learning_starts=0,
            gamma=gamma,
            train_freq=1,
            target_update_interval=targetUpdateInterval,
            exploration_fraction=explorationFraction,
            exploration_initial_eps=initialEpsilon,
            exploration_final_eps=finalEpsilon,
            policy_kwargs=dict(net_arch=[netArch[0], netArch[1]]),
            
            verbose=verbose
        )
        #tmp_path = "./tmp/dqn_log/"
        #new_logger = configure(tmp_path, ["stdout", "csv"])
        #self.model.set_logger(new_logger)
        self.steps_per_episode = env.totalTimeSteps
        
        if callback == None:
            self.callback = CustomMetricsCallback(verbose=1)
        else:   
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
        prev_env = self.model.get_env()
        if prev_env != None and prev_env.observation_space != env.observation_space:
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

    def loadModel(self, modelPath, env):
        model = DQN.load(path=modelPath, env=env)
        self.model = model

        
    def run(self, episodes: int = 1):
        totalAccWaitingTime = 0
        totalAccReward = 0
        
        env=self.model.get_env()
        if env == None:
            raise ValueError("env is not defined.")
        for _ in range(episodes):
            obs = env.reset()
            steps = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
    
                totalAccReward += reward
                totalAccWaitingTime += int(info[0]["mean_acc_waiting_time"])
                steps += 1
                
        meanAccReward = totalAccReward / episodes  
        meanAccWaitingTime = totalAccWaitingTime / steps
        metrics = {"mean_acc_waiting_time": meanAccWaitingTime, "mean_acc_reward": meanAccReward}
        
        return metrics
            