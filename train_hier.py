"""
Main file for training the high-level commander policy.

ATTENTION: Applicable from ray version 2.3.0!!
"""

import os
import time
import shutil
import tqdm
import torch
import logging
import numpy as np
from gymnasium import spaces
from tensorboard import program
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from models.ac_models_hier import CommanderGru
from config import Config
from envs.env_hier import HighLevelEnv

N_OPP_HL = 2
ACT_DIM = N_OPP_HL+1
OBS_DIM = 14+10*N_OPP_HL

def update_logs(args, log_dir):
    """
    Copy stored checkpoints from Ray log to experiment log directory.
    """

    dirs = sorted(Path(log_dir).glob('*/'), key=os.path.getmtime)
    check = ''
    event = ''
    for item in dirs:
        if "checkpoint" in item.name:
            check = str(item)
        if "events" in item.name:
            event = str(item)
    
    result_dir = os.path.join(args.log_path, 'checkpoint')
    
    try:
        shutil.rmtree(result_dir)
    except:
        pass

    shutil.copytree(check,result_dir,symlinks=False,dirs_exist_ok=False)
    shutil.copy(event,result_dir)

def evaluate(args, algo, env, epoch):

    def cc_obs(obs):
        return {
            "obs_1_own": obs,
            "obs_2": np.zeros(OBS_DIM, dtype=np.float32),
            "obs_3": np.zeros(OBS_DIM, dtype=np.float32),
            "act_1_own": np.zeros(1),
            "act_2": np.zeros(1),
            "act_3": np.zeros(1),
        }
    
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    while not done:
        actions = {}
        states = [torch.zeros(200), torch.zeros(200)]

        for ag_id, ag_s in state.items():
            a = algo.compute_single_action(observation=cc_obs(ag_s), state=states, policy_id="commander_policy")
            actions[ag_id] = a[0]
            states[0] = a[1][0]
            states[1] = a[1][1]

        state, rew, hist, trunc, _ = env.step(actions)

        done = hist["__all__"] or trunc["__all__"]
        for r in rew.values():
            reward += r

        step += 1
        if args.render:
            env.plot(Path(args.log_path, "current.png"))
            time.sleep(0.18)

    reward = round(reward, 3)
    env.plot(Path(args.log_path, f"Ep_{epoch}_It_{step}_Rew_{reward}.png"))

def make_checkpoint(args, algo, log_dir, epoch, env=None):
    algo.save()
    update_logs(args, log_dir)
    if args.eval and epoch%100==0:
        for _ in range(2):
            evaluate(args, algo, env, epoch)

def get_policy(args):
    class CustomCallback(DefaultCallbacks):
        """
        Here, the opponent's actions will be added to the episode states 
        And the current level will be tracked. 
        """
        def on_postprocess_trajectory(
            self,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs
        ):
            acts = []

            to_update = postprocessed_batch[SampleBatch.CUR_OBS]
            _, own_batch = original_batches[agent_id]
            own_act = np.squeeze(own_batch[SampleBatch.ACTIONS])
            acts.append(own_act)

            oth_agents = list(range(1,4))
            oth_agents.remove(agent_id)

            for i in oth_agents:
                _, oth_batch = original_batches[i]
                oth_act = np.squeeze(oth_batch[SampleBatch.ACTIONS])
                acts.append(oth_act)
            
            for i, act in enumerate(acts):
                to_update[:,i] = act/3

    def central_critic_observer(agent_obs, **kw):
        """
        Determines which agents will get an observation. 
        In 'on_postprocess_trajectory', the keys will be called lexicographically. 
        """
        new_obs = {
            1: {
                "obs_1_own": agent_obs[1] ,
                "obs_2": agent_obs[2],
                "obs_3": agent_obs[3],
                "act_1_own": np.zeros(1),
                "act_2": np.zeros(1),
                "act_3": np.zeros(1),
            },
            2: {
                "obs_1_own": agent_obs[2] ,
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[3],
                "act_1_own": np.zeros(1),
                "act_2": np.zeros(1),
                "act_3": np.zeros(1),
            },
            3: {
                "obs_1_own": agent_obs[3] ,
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[2],
                "act_1_own": np.zeros(1),
                "act_2": np.zeros(1),
                "act_3": np.zeros(1),
            },
        }
        return new_obs

    ModelCatalog.register_custom_model("commander_model",CommanderGru)
    action_space = spaces.Discrete(ACT_DIM)
    observer_space = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(OBS_DIM,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(OBS_DIM,)),
            "obs_3": spaces.Box(low=0, high=1, shape=(OBS_DIM,)),
            "act_1_own": spaces.Box(low=0, high=ACT_DIM-1, shape=(1,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=ACT_DIM-1, shape=(1,), dtype=np.float32),
            "act_3": spaces.Box(low=0, high=ACT_DIM-1, shape=(1,), dtype=np.float32),
        }
    )

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, batch_mode="complete_episodes", enable_connectors=False) #compare with cetralized_critic_2.py
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=HighLevelEnv, env_config=args.env_config)
        .training(train_batch_size=args.batch_size, gamma=0.99, clip_param=0.25,lr=1e-4, sgd_minibatch_size=args.mini_batch_size)
        .framework("torch")

        # even though only 'one commander agent' is acting, we use multi-agent to have centalized critic option -> CTDE
        .multi_agent(policies={
                "commander_policy": PolicySpec(
                    None,
                    observer_space,
                    action_space,
                    config={
                        "model": {
                            "custom_model": "commander_model"
                        }
                    }
                )
            },
            policy_mapping_fn= lambda agent_id, episode, worker, **kwargs: "commander_policy",
            observation_fn=central_critic_observer)
        .callbacks(CustomCallback)
        .build()
    )
    return algo

if __name__ == '__main__':
    rllib_logger = logging.getLogger("ray.rllib")
    rllib_logger.setLevel(logging.ERROR)
    args = Config(1).get_arguments
    test_env = None
    algo = get_policy(args)

    if args.restore:
        if args.restore_path:
            algo.restore(args.restore_path)
        else:
            algo.restore(os.path.join(args.log_path, "checkpoint"))
    if args.eval:
        test_env = HighLevelEnv(args.env_config)

    log_dir = os.path.normpath(algo.logdir)
    tb = program.TensorBoard()
    port = 6006
    started = False
    while not started:
        try:
            tb.configure(argv=[None, '--logdir', log_dir, '--bind_all', f'--port={port}'])
            url = tb.launch()
            started = True
        except:
            port += 1

    print("\n", "--- NO ERRORS FOUND, STARTING TRAINING ---")

    time.sleep(2)
    time_acc = 0
    iters = tqdm.trange(0, args.epochs+1,  leave=True)
    os.system('clear') if os.name == 'posix' else os.system('cls')

    for i in iters:
        t = time.time()
        result = algo.train()
        time_acc += time.time()-t
        iters.set_description(f"{i}) Reward = {result['episode_reward_mean']:.2f} | Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")
        
        if i % 50 == 0:
            make_checkpoint(args, algo, log_dir, i, test_env)

