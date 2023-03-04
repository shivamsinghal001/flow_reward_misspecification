import importlib
import numpy as np

from flow.core.rewards import REWARD_REGISTRY

class ProxyRewardEnv(object):
    """
    Wraps the given environment in a proxy reward wrapper function that changes the reward provided to the environment.

    Params
    ------
    env: the Flow environment object that will be wrapped with a proxy
    reward_specification: a dict of reward_pairs with str keys corresponding to reward type and float values corresponding to the weight of that reward. 
                            The proxy reward is a linear combination of all the reward functions specified. 
    *args: environment args
    **kwargs: envrionment kwargs
    """
    def __init__(self, module, mod_name, env_params, sim_params, network, simulator,  reward_specification, reward_fun):
        cls = getattr(importlib.import_module(module), mod_name)
        self.env = cls(env_params, sim_params, network, simulator)        
        
        if reward_specification is not None:
            self.use_new_spec = True
            self.reward_fun = reward_fun

            self.true_reward_specification = []
            self.true_disc_action_noise = 0
            self.true_gaussian_noise = 0

            self.obs_reward_specification = []
            self.obs_disc_action_noise = 0
            self.obs_gaussian_noise = 0

            if reward_specification["true"] is not None:
                self.use_original_true = False
                for name, eta in reward_specification["true"]:
                    if name == 'action_noise':
                        assert self.true_gaussian_noise == 0 and self.true_disc_action_noise == 0
                        self.true_gaussian_noise = eta
                    elif name == 'disc_action_noise':
                        assert self.true_disc_action_noise == 0 and self.true_gaussian_noise == 0
                        self.true_disc_action_noise = eta
                    else:
                        assert name in REWARD_REGISTRY 
                        self.true_reward_specification.append((REWARD_REGISTRY[name], eta))
            else:
                self.use_original_true = True  
                self.original_rew_func = getattr(self.env, "compute_reward")  

            for name, eta in reward_specification["observed"]:
                if name == 'action_noise':
                    assert self.obs_gaussian_noise == 0 and self.obs_disc_action_noise == 0
                    self.obs_gaussian_noise = eta
                elif name == 'disc_action_noise':
                    assert self.obs_disc_action_noise == 0 and self.obs_gaussian_noise == 0
                    self.obs_disc_action_noise = eta
                else:
                    assert name in REWARD_REGISTRY 
                    self.obs_reward_specification.append((REWARD_REGISTRY[name], eta))
            
            if self.reward_fun == "observed":
                def proxy_reward(rl_actions, **kwargs):
                    return self._proxy(self.obs_reward_specification, rl_actions, **kwargs)
                setattr(self.env, "compute_reward", proxy_reward)
            elif not self.use_original_true:
                def proxy_reward(rl_actions, **kwargs):
                    return self._proxy(self.true_reward_specification, rl_actions, **kwargs)             
                setattr(self.env, "compute_reward", proxy_reward)
        else:
            self.use_new_spec = False

    def _proxy(self, reward_specification, rl_actions, **kwargs):
        vel = np.array(self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))
        if any(vel < -100) or kwargs["fail"]:
            return 0
        rew = 0 
        for fn, eta in reward_specification:
            rew += eta * fn(self.env, rl_actions)
        return rew 
        
    def __getattr__(self, attr):
        return self.env.__getattribute__(attr)

    def step(self, rl_actions):
        next_observation, reward, done, infos = self.env.step(rl_actions)
        if self.use_new_spec:
            if self.reward_fun == "observed":
                infos["observed_reward"] = reward
                if not self.use_original_true:
                    infos["true_reward"] = self._proxy(self.true_reward_specification, rl_actions, fail=infos["crash"])
                else:
                    infos["true_reward"] = self.original_rew_func(rl_actions, fail=infos["crash"])
            else:
                infos["observed_reward"] = self._proxy(self.obs_reward_specification, rl_actions, fail=infos["crash"])
                infos["true_reward"] = reward
        else:
            infos["observed_reward"] = infos["true_reward"] = reward
        return next_observation, reward, done, infos

    def _apply_rl_actions(self, rl_actions):
        if self.use_new_spec:
            if self.reward_fun == "observed":
                if self.obs_disc_action_noise != 0:
                    # round to the noise level
                    self.env._apply_rl_actions(
                        np.round(rl_actions / self.obs_disc_action_noise) * self.obs_disc_action_noise
                    )
                else:
                    self.env._apply_rl_actions(
                        rl_actions + np.random.normal(scale=self.obs_gaussian_noise, size=len(rl_actions))
                    )
            elif not self.use_original_true:
                if self.true_disc_action_noise != 0:
                    # round to the noise level
                    self.env._apply_rl_actions(
                        np.round(rl_actions / self.true_disc_action_noise) * self.true_disc_action_noise
                    )
                else:
                    self.env._apply_rl_actions(
                        rl_actions + np.random.normal(scale=self.true_gaussian_noise, size=len(rl_actions))
                    )
        else:
            self.env._apply_rl_actions(rl_actions)

