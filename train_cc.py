"""
Fully Centralized Critic Q(s,a) where s and a contain all states and actions.

ATTENTION: Ray works in this way until version 2.2.0. From version 2.3.0, it's different.
"""

import os
import time
import shutil
import tqdm
import json
import random
import numpy as np
from gym import spaces
from tensorboard import program
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from models.ac_models_cc import CCRocket, CCRocketDummy, CCRocketFri
from models.ac_models_rnn import CCRocketLstm,MsgPassBi, CCGRUSkip
from config_cc import Config
from envs.env_warsim_cc import DogfightScenario


ACTION_DIM = 4

def update_logs(args, log_dir, level, epoch):
    dirs = sorted(Path(log_dir).glob('*/'), key=os.path.getmtime)
    check = ''
    event = ''
    for item in dirs:
        if "checkpoint" in item.name:
            check = str(item)
        if "events" in item.name:
            event = str(item)
    
    if args.exp_folder:
        result_dir = os.path.join("results", args.exp_folder, args.log_name, 'checkpoint')
    else:
        result_dir = os.path.join("results", args.log_name, 'checkpoint')
    
    try:
        shutil.rmtree(result_dir)
    except:
        pass

    shutil.copytree(check,result_dir,symlinks=False,dirs_exist_ok=False)
    shutil.copy(event,result_dir)
    if args.log_history and epoch % 200 == 0:
        history_dir = os.path.join("results", args.log_name, 'check_history',f'check_{epoch}_{level}')
        shutil.copytree(check,history_dir,symlinks=False,dirs_exist_ok=False)
        shutil.copy(event,history_dir)

def evaluate(args, algo, env, epoch, level):
    def cc_obs(obs, id):
        if id == 1:
            return {
                "obs_1_own": obs[1] ,
                "obs_2": obs[2],
                "obs_3": obs[3],
                "obs_4": obs[4],
                "act_1_own": np.zeros(ACTION_DIM),
                "act_2": np.zeros(ACTION_DIM),
                "act_3": np.zeros(ACTION_DIM),
                "act_4": np.zeros(ACTION_DIM),
            }
        elif id == 2:
            return {
                "obs_1_own": obs[2] ,
                "obs_2": obs[1],
                "obs_3": obs[3],
                "obs_4": obs[4],
                "act_1_own": np.zeros(ACTION_DIM),
                "act_2": np.zeros(ACTION_DIM),
                "act_3": np.zeros(ACTION_DIM),
                "act_4": np.zeros(ACTION_DIM),
            }
    
    state = env.reset()
    reward = 0
    done = False
    step = 0
    while not done:
        actions = {}
        for ag_id, ag_s in state.items():
            if ag_id <= 2:
                actions[ag_id] = algo.compute_single_action(observation=cc_obs(state, ag_id), policy_id=f"{args.agent_mode}_policy")

        state, rew, hist, _ = env.step(actions)
        done = hist["__all__"]
        for r in rew.values():
            reward += r

        step += 1
        if args.render:
            env.plot(Path(args.log_path, "current.png"))
            time.sleep(0.18)

    reward = round(reward, 3)
    env.plot(Path(args.log_path, f"Ep_{epoch}_It_{step}_Lv{level}_Rew_{reward}.png"))

def make_checkpoint(args, algo, log_dir, epoch, level, env=None):
    algo.save()
    update_logs(args, log_dir, level, epoch)
    if args.eval and epoch%200==0:
        for _ in range(2):
            evaluate(args, algo, env, epoch, level)

def cc_policy(args):

    class CustomCallback(DefaultCallbacks):
        """
        Here, the opponents actions will be added to the episode states 
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
            to_update = postprocessed_batch[SampleBatch.CUR_OBS]
            other_id = 2 if agent_id == 1 else 1
            _, own_batch = original_batches[agent_id]
            own_act = np.squeeze(own_batch[SampleBatch.ACTIONS])
            _, fri_batch = original_batches[other_id]
            fri_act = np.squeeze(fri_batch[SampleBatch.ACTIONS])
            opp_acts = postprocessed_batch[SampleBatch.INFOS] # list with dicts {3:[act], 4:[act]}
            
            acts = [own_act, fri_act, opp_acts]

            for i, act in enumerate(acts):
                if i <= 1:
                    to_update[:,i*4] = act[:,0]/12.0
                    to_update[:,i*4+1] = act[:,1]/8.0
                    to_update[:,i*4+2] = act[:,2]
                    to_update[:,i*4+3] = act[:,3]
                else:
                    act3 = np.zeros((len(act),4), dtype=np.float32)
                    act4 = np.zeros((len(act),4), dtype=np.float32)
                    for j, ac in enumerate(act):
                        a3 = ac[3] #act list opp3
                        a4 = ac[4] #act list opp4

                        act3[j,0] = a3[0]/12.0
                        act3[j,1] = a3[1]/8.0
                        act3[j,2] = a3[2]
                        act3[j,3] = a3[3]

                        act4[j,0] = a4[0]/12.0
                        act4[j,1] = a4[1]/8.0
                        act4[j,2] = a4[2]
                        act4[j,3] = a4[3]

                    to_update[:, 8:12] = act3
                    to_update[:, 12:16] = act4

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
                "obs_4": agent_obs[4],
                "act_1_own": np.zeros(ACTION_DIM),
                "act_2": np.zeros(ACTION_DIM),
                "act_3": np.zeros(ACTION_DIM),
                "act_4": np.zeros(ACTION_DIM),
            },
            2: {
                "obs_1_own": agent_obs[2] ,
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[3],
                "obs_4": agent_obs[4],
                "act_1_own": np.zeros(ACTION_DIM),
                "act_2": np.zeros(ACTION_DIM),
                "act_3": np.zeros(ACTION_DIM),
                "act_4": np.zeros(ACTION_DIM),
            }
        }
        return new_obs

    observer_space = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(args.obs_dim,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(args.obs_dim,)),
            "obs_3": spaces.Box(low=0, high=1, shape=(args.obs_dim,)),
            "obs_4": spaces.Box(low=0, high=1, shape=(args.obs_dim,)),
            "act_1_own": spaces.Box(low=0, high=12, shape=(ACTION_DIM,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=12, shape=(ACTION_DIM,), dtype=np.float32),
            "act_3": spaces.Box(low=0, high=12, shape=(ACTION_DIM,), dtype=np.float32),
            "act_4": spaces.Box(low=0, high=12, shape=(ACTION_DIM,), dtype=np.float32),
        }
    )

    ModelCatalog.register_custom_model(f"{args.agent_mode}_model",CCGRUSkip)
    ModelCatalog.register_custom_model("dummy_model",CCRocketDummy)
    action_space = spaces.MultiDiscrete([13,9,2,2])

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, horizon=args.horizon, batch_mode="complete_episodes")
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=DogfightScenario, env_config=args.env_config)
        .training(train_batch_size=args.batch_size, gamma=0.99, clip_param=0.25,lr=1e-4, lambda_=0.95, sgd_minibatch_size=256)
        .framework("torch")
        .multi_agent(policies={
                f"{args.agent_mode}_policy": PolicySpec(
                    None,
                    observer_space,
                    action_space,
                    config={
                        "model": {
                            "custom_model": f"{args.agent_mode}_model"
                        }
                    }
                ),
                "dummy_policy": PolicySpec(
                    None,
                    observer_space,
                    action_space,
                    config={
                        "model": {
                            "custom_model": "dummy_model"
                        }
                    }
                )
            },
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: f"{args.agent_mode}_policy" if agent_id <=2 else "dummy_policy"),
            policies_to_train=[f"{args.agent_mode}_policy"],
            observation_fn=central_critic_observer)
        .callbacks(CustomCallback)
        .build()
    )
    return algo

if __name__ == '__main__':
    args = Config().get_arguments
    test_env = None
    algo = cc_policy(args)

    if args.restore:
        if args.restore_path:
            algo.restore(args.restore_path)
        else:
            algo.restore(os.path.join(args.log_path, "checkpoint"))
    if args.eval:
        test_env = DogfightScenario(args.env_config)

    log_dir = os.path.normpath(algo.logdir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()

    print("\n", "--- NO ERRORS FOUND, STARTING TRAINING ---")

    time.sleep(2)
    time_acc = 0
    iters = tqdm.trange(0, 10*args.epochs+1,  leave=True)
    os.system('clear') if os.name == 'posix' else os.system('cls')

    for i in iters:
        t = time.time()
        result = algo.train()
        time_acc += time.time()-t
        #iters.set_description(f"{i}) Reward = {result['episode_reward_mean']:.2f} | Level = {level} | Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")
        iters.set_description(f"{i}) Reward = {result['episode_reward_mean']:.2f} | Mode = {args.agent_mode}, AC{args.ac_type}, Level = {args.level} | Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")

        if i % 50 == 0:
            make_checkpoint(args, algo, log_dir, i, args.level, test_env)

