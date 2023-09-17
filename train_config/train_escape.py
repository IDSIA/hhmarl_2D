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
from models.escape_mod import EscapeOne
from train_config.config_escape import Config
from backup_files.envs.env_escape import DogfightScenario

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
    
    result_dir = os.path.join("results", args.log_name, 'checkpoint')
    history_dir = os.path.join("results", args.log_name, 'check_history',f'check_{epoch}_{level}')
    try:
        shutil.rmtree(result_dir)
    except:
        pass

    shutil.copytree(check,result_dir,symlinks=False,dirs_exist_ok=False)
    shutil.copy(event,result_dir)
    if epoch % 200 == 0:
        shutil.copytree(check,history_dir,symlinks=False,dirs_exist_ok=False)
        shutil.copy(event,history_dir)

def evaluate(args, algo, env, epoch, level):
    def cc_obs(obs, id):
        if id == 1:
            return {
                "own_obs": obs[1],
                "opponent_obs": obs[2],
                "opponent_action": np.zeros(ACTION_DIM)
            }
        elif id == 2:
            return {
                "own_obs": obs[2],
                "opponent_obs": obs[1],
                "opponent_action": np.zeros(ACTION_DIM)
            }
    
    state = env.reset()
    reward = 0
    done = False
    step = 0
    while not done:
        actions = {}
        for ag_id, ag_s in state.items():
            actions[ag_id] = algo.compute_single_action(observation=cc_obs(state, ag_id), policy_id="cc_policy")

        state, rew, hist, _ = env.step(actions)
        done = hist["__all__"]
        for r in rew.values():
            reward += r

        step += 1
        if args.render:
            env.plot(Path("results", args.log_name, "current.png"))
            time.sleep(0.18)

    reward = round(reward, 3)
    env.plot(Path("results", args.log_name, f"Ep_{epoch}_It_{step}_Lv{level}_Rew_{reward}.png"))

def make_checkpoint(args, algo, log_dir, epoch, level, env=None):
    algo.save()
    update_logs(args, log_dir, level, epoch)
    if args.eval and epoch%100==0:
        for _ in range(1):
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
            _, opponent_batch = original_batches[other_id]
            try:
                opponent_action = np.squeeze(opponent_batch[SampleBatch.ACTIONS])
            except:
                opponent_action = None
            if opponent_action is not None:
                if opponent_action.ndim == 1:
                    a = round(float(opponent_action[0] / 12.0), 3)
                    b = round(float(opponent_action[1] / 8.0), 3)
                    c = round(float(opponent_action[2]), 3)
                    d = round(float(opponent_action[3]), 3)
                    to_update[:, :ACTION_DIM] = np.array([a,b,c,d])
                else:
                    a = opponent_action[:,0]
                    a_val = np.zeros((len(a)), dtype=np.float32)
                    for i,v in enumerate(a):
                        a_val[i] = round(float(v)/12.0,3)
                    b = opponent_action[:,1]
                    b_val = np.zeros((len(b)), dtype=np.float32)
                    for i,v in enumerate(b):
                        b_val[i] = round(float(v/8),3)
                    c = opponent_action[:,2]
                    d = opponent_action[:,3]
                    to_update[:,0] = a_val
                    to_update[:,1] = b_val
                    to_update[:,2] = c
                    to_update[:,3] = d

    def central_critic_observer(agent_obs, **kw):
        new_obs = {
            1: {
                "own_obs": agent_obs[1] ,
                "opponent_obs": agent_obs[2],
                "opponent_action": np.zeros(ACTION_DIM),
            },
            2: {
                "own_obs": agent_obs[2],
                "opponent_obs": agent_obs[1],
                "opponent_action": np.zeros(ACTION_DIM),
            },
        }
        return new_obs

    observer_space = spaces.Dict(
        {
            "own_obs": spaces.Box(low=0, high=1, shape=(args.obs_dim,)),
            "opponent_obs": spaces.Box(low=0, high=1, shape=(args.obs_dim,)),
            "opponent_action": spaces.Box(low=0, high=12, shape=(ACTION_DIM,), dtype=np.float32),
        }
    )

    ModelCatalog.register_custom_model("cc_model",EscapeOne)
    action_space = spaces.MultiDiscrete([13,9,2,2])

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, horizon=args.horizon, batch_mode="complete_episodes")
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=DogfightScenario, env_config=args.env_config)
        .training(train_batch_size=args.batch_size, gamma=0.99, clip_param=0.25,lr=1e-4, lambda_=0.95)
        .framework("torch")
        .multi_agent(policies={
                "cc_policy": PolicySpec(
                    None,
                    observer_space,
                    action_space,
                    config={
                        "model": {
                            "custom_model": "cc_model"
                        }
                    }
                )
            },
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "cc_policy"),
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

    time.sleep(1)
    time_acc = 0
    iters = tqdm.trange(0, args.epochs+1,  leave=True)
    os.system('clear') if os.name == 'posix' else os.system('cls')

    for i in iters:
        t = time.time()
        result = algo.train()
        time_acc += time.time()-t
        iters.set_description(f"{i}) Reward = {result['episode_reward_mean']:.2f} | Level = {args.level} | Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")

        if i % 100 == 0:
            make_checkpoint(args, algo, log_dir, i, args.level, test_env)

