"""
Main file for training low-level heterogeneous agents.

ATTENTION: Applicable from ray version 2.3.0!!

HETEROGENEOUS: Agend IDs to AC types: 1->1, 2->2, 3->1, 4->2

"""

import os
import time
import shutil
import tqdm
import numpy as np
from gymnasium import spaces
from tensorboard import program
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from models.ac_models_hetero import Fight1, Fight2, Esc1, Esc2, DummyFight1, DummyFight2, DummyEsc1, DummyEsc2
from config_hetero import Config
from envs.env_hetero import DogfightScenario
import torch


ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3

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

def evaluate(args, algo, env, epoch, level):
    """
    Evaluations are stored as pictures of combat scenarios, with rewards in filename.
    """
    def cc_obs(obs, id):
        if id == 1:
            return {
                "obs_1_own": obs[1] ,
                "obs_2": obs[2],
                "obs_3": obs[3],
                "obs_4": obs[4],
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
                "act_3": np.zeros(ACTION_DIM_AC1),
                "act_4": np.zeros(ACTION_DIM_AC2),
            }
        elif id == 2:
            return {
                "obs_1_own": obs[2] ,
                "obs_2": obs[1],
                "obs_3": obs[3],
                "obs_4": obs[4],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
                "act_3": np.zeros(ACTION_DIM_AC1),
                "act_4": np.zeros(ACTION_DIM_AC2),
            }
    
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    while not done:
        actions = {}
        for ag_id in state.keys():
            if ag_id <= 2:
                a = algo.compute_single_action(observation=cc_obs(state, ag_id), state=torch.zeros(1), policy_id=f"ac{ag_id}_policy")
                actions[ag_id] = a[0]

        state, rew, hist, trunc, _ = env.step(actions)
        done = hist["__all__"] or trunc["__all__"]
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
    if args.eval and epoch%100==0:
        for _ in range(2):
            evaluate(args, algo, env, epoch, level)

def assign_policy(agent_id, episode, worker, **kwargs):
    if agent_id == 1:
        return "ac1_policy"
    elif agent_id == 2:
        return "ac2_policy"
    elif agent_id == 3:
        return "dummy_policy_ac1"
    elif agent_id == 4:
        return "dummy_policy_ac2"

def get_policy(args):
    """
    Agents get assigned the neural networks CCAtt1, CCAtt2, CCEsc1 and CCEsc2. 
    Opponents need to get assigned also a network to have them registered as 'agents' in Ray. This is needed to get their IDs in callbacks.
    We assign them a dummy network, which won't be used and updated.
    """

    class CustomCallback(DefaultCallbacks):
        """
        This callback is used to have fully observable critic. Other agent's and opponent's
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
            opp_acts = postprocessed_batch[SampleBatch.INFOS] # list with dicts {3:[act], 4:[act]}
            
            acts = [own_act, fri_act, opp_acts]

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
                    
                else:
                    act3 = np.zeros((len(act),4), dtype=np.float32)
                    act4 = np.zeros((len(act),3), dtype=np.float32)
                    for j, ac in enumerate(act):
                        try:
                            ac = ac[agent_id]
                        except:
                            pass
                        a3 = ac[3] #act list opp3
                        a4 = ac[4] #act list opp4

                        act3[j,0] = a3[0]/12.0
                        act3[j,1] = a3[1]/8.0
                        act3[j,2] = a3[2]
                        act3[j,3] = a3[3]

                        act4[j,0] = a4[0]/12.0
                        act4[j,1] = a4[1]/8.0
                        act4[j,2] = a4[2]

                    to_update[:, 7:11] = act3
                    to_update[:, 11:14] = act4

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
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
                "act_3": np.zeros(ACTION_DIM_AC1),
                "act_4": np.zeros(ACTION_DIM_AC2),
            },
            2: {
                "obs_1_own": agent_obs[2] ,
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[3],
                "obs_4": agent_obs[4],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
                "act_3": np.zeros(ACTION_DIM_AC1),
                "act_4": np.zeros(ACTION_DIM_AC2),
            }
        }
        return new_obs

    observer_space_ac1 = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(30,)),
            "obs_2": spaces.Box(low=0, high=1, shape=(28+int(args.agent_mode=="escape"),)),
            "obs_3": spaces.Box(low=0, high=1, shape=(30,)),
            "obs_4": spaces.Box(low=0, high=1, shape=(28+int(args.agent_mode=="escape"),)),
            "act_1_own": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
            "act_3": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
            "act_4": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
        }
    )
    observer_space_ac2 = spaces.Dict(
        {
            "obs_1_own": spaces.Box(low=0, high=1, shape=(28+int(args.agent_mode=="escape"),)),
            "obs_2": spaces.Box(low=0, high=1, shape=(30,)),
            "obs_3": spaces.Box(low=0, high=1, shape=(30,)),
            "obs_4": spaces.Box(low=0, high=1, shape=(28+int(args.agent_mode=="escape"),)),
            "act_1_own": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
            "act_2": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
            "act_3": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC1,), dtype=np.float32),
            "act_4": spaces.Box(low=0, high=12, shape=(ACTION_DIM_AC2,), dtype=np.float32),
        }
    )

    if args.agent_mode == "escape":
        ModelCatalog.register_custom_model("ac1_model",Esc1)
        ModelCatalog.register_custom_model("ac2_model",Esc2)
        ModelCatalog.register_custom_model("dummy_model_ac1",DummyEsc1)
        ModelCatalog.register_custom_model("dummy_model_ac2",DummyEsc2)
    else:
        ModelCatalog.register_custom_model("ac1_model_l5" if args.level == 5 else 'ac1_model', Fight1) # 'ac1_model' until level4, ac1_model_l5 at level5 because of esc policy registration
        ModelCatalog.register_custom_model("ac2_model_l5" if args.level == 5 else 'ac2_model',Fight2)
        ModelCatalog.register_custom_model("dummy_model_ac1",DummyFight1)
        ModelCatalog.register_custom_model("dummy_model_ac2",DummyFight2)

    action_space_ac1 = spaces.MultiDiscrete([13,9,2,2])
    action_space_ac2 = spaces.MultiDiscrete([13,9,2])

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, batch_mode="complete_episodes", enable_connectors=False) #compare with cetralized_critic_2.py
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=DogfightScenario, env_config=args.env_config)
        .training(train_batch_size=args.batch_size, gamma=0.99, clip_param=0.25,lr=1e-4, lambda_=0.95, sgd_minibatch_size=256)
        .framework("torch")
        .multi_agent(policies={
                "ac1_policy": PolicySpec(
                    None,
                    observer_space_ac1,
                    action_space_ac1,
                    config={
                        "model": {
                            "custom_model": "ac1_model_l5" if args.level == 5 else 'ac1_model'
                        }
                    }
                ),
                "ac2_policy": PolicySpec(
                    None,
                    observer_space_ac2,
                    action_space_ac2,
                    config={
                        "model": {
                            "custom_model": "ac2_model_l5" if args.level == 5 else 'ac2_model'
                        }
                    }
                ),
                "dummy_policy_ac1": PolicySpec(
                    None,
                    observer_space_ac1,
                    action_space_ac1,
                    config={
                        "model": {
                            "custom_model": "dummy_model_ac1"
                        }
                    }
                ),
                "dummy_policy_ac2": PolicySpec(
                    None,
                    observer_space_ac2,
                    action_space_ac2,
                    config={
                        "model": {
                            "custom_model": "dummy_model_ac2"
                        }
                    }
                )
            },
            policy_mapping_fn=assign_policy,
            policies_to_train=["ac1_policy", "ac2_policy"],
            observation_fn=central_critic_observer)
        .callbacks(CustomCallback)
        .build()
    )
    return algo

if __name__ == '__main__':
    args = Config().get_arguments
    test_env = None
    algo = get_policy(args)

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
    #tb.configure(argv=[None, '--logdir', log_dir, '--port=6007'])
    url = tb.launch()

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

