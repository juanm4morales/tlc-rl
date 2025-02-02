from time import time
from collections import deque
import numpy as np
import optuna
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class CustomMetricsCallback(BaseCallback):
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        #self.total_acc_waiting_times = 0
        self.total_losses = 0
        self.cumulative_reward = 0
        self.step = 0
        self.episode = 0
        self.arrivalCWT = []
        self.metrics = {"episode": [], "mean_acc_waiting_time": [], "cumulative_reward":[], "time": [], "loss_value": [], "waiting_time_95p":[]}
        self.episode_start_time = 0
        
    def _on_training_start(self) -> None:
        self.episode_start_time = time()

    def _on_step(self) -> bool:
        #mean_acc_waiting_time_step = self.locals['infos'][0]['mean_acc_waiting_time']
        self.arrivalCWT += self.locals['infos'][0]['arrival_acc_waiting_times']
        #self.total_acc_waiting_times += mean_acc_waiting_time_step
        
        done = self.locals['dones'][0]
        self.cumulative_reward += self.locals['rewards'][0]
        
        self.step += 1
        if done:
            self._on_episode_end()
            
        return True
    
    def _on_episode_end(self):
        self.metrics["episode"].append(self.episode)
        #mean_acc_waiting_time_ep = self.total_acc_waiting_times / self.step
        #self.metrics["mean_acc_waiting_time"].append(mean_acc_waiting_time_ep)
        self.metrics["cumulative_reward"].append(self.cumulative_reward)
        episode_end_time = time()
        episode_time = episode_end_time - self.episode_start_time
        self.metrics["time"].append(episode_time)
        
        self.metrics["loss_value"].append(self.model.logger.name_to_value.get('train/loss'))
        
        mean_cwt = np.mean(self.arrivalCWT)
        p95_cwt = np.percentile(self.arrivalCWT, 95)
        self.metrics["mean_acc_waiting_time"].append(mean_cwt)
        self.metrics["waiting_time_95p"].append(p95_cwt)
        
        self.step = 0
        #self.total_acc_waiting_times = 0
        self.cumulative_reward = 0
        self.arrivalCWT = []
        self.episode += 1

        self.logger.record("train/mean_acc_waiting_time", mean_cwt)
        self.logger.record("train/95p_acc_waiting_time", p95_cwt)
        self.logger.record("time/episode_time", episode_time)
        
        self.episode_start_time = time()
        
    def get_metrics(self):
        return self.metrics
    

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
    
class TrialCallback(BaseCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        trial: optuna.Trial,
        n_eval_episodes: int,
        rewards_window_size: int,
        min_trial_fract: float,
        verbose: int = 0,
        max_trial_time: int = 2700,
        prune: bool = True
    ):
        super().__init__(verbose)
        self.trial = trial
        self.episode = 0
        self.cumulative_reward = 0
        self.last_cumulative_rewards = deque(maxlen=rewards_window_size)
        self.prune = prune
        self.is_pruned = False
        assert(min_trial_fract>=0 and min_trial_fract<=1)
        self.min_episodes_trial = int(min_trial_fract * n_eval_episodes)
        self.max_trial_time = max_trial_time
        self.initial_time = time()

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        self.cumulative_reward += self.locals['rewards'][0]
        
        if done:
            self._on_episode_end()
            self.episode += 1
            self.cumulative_reward = 0
        return True
    
    def _on_episode_end(self):
        self.last_cumulative_rewards.append(self.cumulative_reward)
        mean_last_cr = sum(self.last_cumulative_rewards)/len(self.last_cumulative_rewards)
        if (self.verbose):
            print("Trial " + str(self.trial.number) + " - Episode " + str(self.episode) + " finished" )
            
        self.trial.report(mean_last_cr, self.episode)
        if self.prune:
            if (time()-self.initial_time) > self.max_trial_time:
                self.is_pruned = True
                return False
            if (self.episode >= self.min_episodes_trial):
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
        
        

