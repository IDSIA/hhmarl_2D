"""
Main file for training low-level heterogeneous agents.

ATTENTION: Applicable from ray version 2.3.0!!

HETEROGENEOUS: Agend IDs to AC types: 1->1, 2->2, 3->1, 4->2

"""

import os
import time
import shutil
import tqdm
import torch
import numpy as np
from gymnasium import spaces
from tensorboard import program
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from models.ac_models_hetero import Esc1, Esc2, Fight1, Fight2
from config import Config
from envs.env_hetero import LowLevelEnv

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

def update_logs(args, log_dir, level, epoch):
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

def evaluate(args, algo, env, epoch, level, it):
    """
    Evaluations are stored as pictures of combat scenarios, with rewards in filename.
    """
    def cc_obs(obs, id):
        if id == 1:
            return {
                "obs_1_own": obs[1] ,
                "obs_2": obs[2],
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
            }
        elif id == 2:
            return {
                "obs_1_own": obs[2] ,
                "obs_2": obs[1],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
            }
    
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    while not done:
        actions = {}
        for ag_id in state.keys():
            a = algo.compute_single_action(observation=cc_obs(state, ag_id), state=torch.zeros(1), policy_id=f"ac{ag_id}_policy", explore=False)
            actions[ag_id] = a[0]

        state, rew, term, trunc, _ = env.step(actions)
        done = term["__all__"] or trunc["__all__"]
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
    for it in range(2):
        if args.level >= 3:
            algo.export_policy_model(os.path.join(os.path.dirname(__file__), 'policies'), f'ac{it+1}_policy')
            policy_name = f'L{args.level}_AC{it+1}' if args.agent_mode == "fight" else f'Esc_AC{it+1}'
            os.rename('policies/model.pt', f'policies/{policy_name}.pt')
        if args.eval and epoch%200==0:
            evaluate(args, algo, env, epoch, level, it)

def get_policy(args):
    """
    Agents get assigned the neural networks Fight1, Fight2 and Esc1, Esc2.
    """
    class CustomCallback(DefaultCallbacks):
        """
        This callback is used to have fully observable critic. Other agent's
        observations and actions will be added to this episode batch.

        ATTENTION: This callback is set up for 2vs2 training.  
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
            
            acts = [own_act, fri_act]

            for i, act in enumerate(acts):
                if agent_id == 1 and i == 0 or agent_id==2 and i==1:
                    if agent_id == 1:
                        to_update[:,i*4] = act[:,0]/12.0
                        to_update[:,i*4+1] = act[:,1]/8.0
                        to_update[:,i*4+2] = act[:,2]
                        to_update[:,i*4+3] = act[:,3]
                    else:
                        to_update[:,i*3] = act[:,0]/12.0
                        to_update[:,i*3+1] = act[:,1]/8.0
                        to_update[:,i*3+2] = act[:,2]
                        to_update[:,i*3+3] = act[:,3]
                elif agent_id == 1 and i == 1 or agent_id==2 and i==0:
                    if agent_id==1:
                        to_update[:,i*4] = act[:,0]/12.0
                        to_update[:,i*4+1] = act[:,1]/8.0
                        to_update[:,i*4+2] = act[:,2]
                    else:
                        to_update[:,i] = act[:,0]/12.0
                        to_update[:,i+1] = act[:,1]/8.0
                        to_update[:,i+2] = act[:,2]

    def central_critic_observer(agent_obs, **kw):
        """
        Determines which agents will get an observation. 
        In 'on_postprocess_trajectory', the keys will be called lexicographically. 
        """
        new_obs = {
            1: {
                "obs_1_own": agent_obs[1] ,
                "obs_2": agent_obs[2],
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
            },
            2: {
                "obs_1_own": agent_obs[2] ,
                "obs_2": agent_obs[1],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
            }
        }
        return new_obs

    observer_space_ac1 = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(OBS_AC1 if args.agent_mode=="fight" else OBS_ESC_AC1,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(OBS_AC2 if args.agent_mode=="fight" else OBS_ESC_AC2,)),
            "act_1_own": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
        }
    )
    observer_space_ac2 = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(OBS_AC2 if args.agent_mode=="fight" else OBS_ESC_AC2,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(OBS_AC1 if args.agent_mode=="fight" else OBS_ESC_AC1,)),
            "act_1_own": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
        }
    )

    if args.agent_mode == "escape":
        ModelCatalog.register_custom_model("ac1_model_esc",Esc1)
        ModelCatalog.register_custom_model("ac2_model_esc",Esc2)
    else:
        ModelCatalog.register_custom_model('ac1_model', Fight1) 
        ModelCatalog.register_custom_model('ac2_model', Fight2)

    action_space_ac1 = spaces.MultiDiscrete([13,9,2,2])
    action_space_ac2 = spaces.MultiDiscrete([13,9,2])

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, batch_mode="complete_episodes", enable_connectors=False) #compare with cetralized_critic_2.py
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=LowLevelEnv, env_config=args.env_config)
        .training(train_batch_size=args.batch_size, gamma=0.99, clip_param=0.25,lr=1e-4, lambda_=0.95, sgd_minibatch_size=args.mini_batch_size)
        .framework("torch")
        .multi_agent(policies={
                "ac1_policy": PolicySpec(
                    None,
                    observer_space_ac1,
                    action_space_ac1,
                    config={
                        "model": {
                            "custom_model": "ac1_model_esc" if args.agent_mode=="escape" else 'ac1_model'
                        }
                    }
                ),
                "ac2_policy": PolicySpec(
                    None,
                    observer_space_ac2,
                    action_space_ac2,
                    config={
                        "model": {
                            "custom_model": "ac2_model_esc" if args.agent_mode=="escape" else 'ac2_model'
                        }
                    }
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f'ac{agent_id}_policy',
            policies_to_train=["ac1_policy", "ac2_policy"],
            observation_fn=central_critic_observer)
        .callbacks(CustomCallback)
        .build()
    )
    return algo

if __name__ == '__main__':
    args = Config(0).get_arguments
    test_env = None
    algo = get_policy(args)

    if args.restore:
        if args.restore_path:
            algo.restore(args.restore_path)
        else:
            algo.restore(os.path.join(args.log_path, "checkpoint"))
    if args.eval:
        test_env = LowLevelEnv(args.env_config)

    log_dir = os.path.normpath(algo.logdir)
    tb = program.TensorBoard()
    port = 6006
    started = False
    url = None
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
        iters.set_description(f"{i}) Reward = {result['episode_reward_mean']:.2f} | Level = {args.level} | Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")
        
        if i % 50 == 0:
            make_checkpoint(args, algo, log_dir, i, args.level, test_env)