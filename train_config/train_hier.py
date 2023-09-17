import os
import time
import shutil
import tqdm
import numpy as np
from gym import spaces
from tensorboard import program
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from train_config.config_hier import Config
from envs.env_hier import DogfightHierarchy

def update_logs(args, log_dir):
    dirs = sorted(Path(log_dir).glob('*/'), key=os.path.getmtime)
    check = ''
    event = ''
    for item in dirs:
        if "checkpoint" in item.name:
            check = str(item)
        if "events" in item.name:
            event = str(item)
    
    result_dir = os.path.join("results", args.log_name, 'checkpoint')
    try:
        shutil.rmtree(result_dir)
    except:
        pass

    shutil.copytree(check,result_dir,symlinks=False,dirs_exist_ok=False)
    shutil.copy(event,result_dir)

def evaluate(args, algo, env, epoch):
    
    state = env.reset()
    reward = 0
    done = False
    step = 0
    while not done:

        actions = algo.compute_single_action(observation=state)
        state, rew, done, _ = env.step(actions)
        reward += rew
        step += 1

        if args.render or args.eval_only:
            env.plot(Path("results", args.log_name, "current.png"))
            time.sleep(0.5)

    reward = round(reward, 3)
    env.plot(Path("results", args.log_name, f"Epoch_{epoch}_Step_{step}_Reward_{reward}.png"))

def make_checkpoint(args, algo, log_dir, epoch, env=None):
    algo.save()
    update_logs(args, log_dir)
    if args.eval and epoch % 400 == 0:
        for i in range(2):
            evaluate(args, algo, env, epoch)

def get_policy(args):
    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=args.num_workers, horizon=args.horizon, batch_mode="complete_episodes")
        .resources(num_gpus=args.gpu)
        .evaluation(evaluation_interval=None)
        .environment(env=DogfightHierarchy, env_config=args.env_config)
        .framework("torch")
        .training(train_batch_size=args.batch_size, gamma=0.99, clip_param=0.2,lr=1e-4, lambda_=0.8)
        .exploration(explore= not args.eval_only)
        .build()
    )
    return algo

if __name__ == '__main__':     
    args = Config().get_arguments
    test_env = DogfightHierarchy(args.env_config)

    if not os.path.exists(os.path.join('results', args.log_name, 'checkpoint')):
        os.makedirs(os.path.join('results', args.log_name, 'checkpoint'))

    algo = get_policy(args)

    if args.restore or args.eval_only:
        algo.restore(args.restore_path)

    log_dir = os.path.normpath(algo.logdir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()

    print("\n", "--- NO ERRORS FOUND, STARTING TRAINING ---")

    time.sleep(1)
    time_acc = 0
    iters = tqdm.trange(0, args.epochs+1,  leave=True)
    os.system('clear') if os.name == 'posix' else os.system('cls')

    if args.eval_only:
        for i in range(10):
            evaluate(args, algo, test_env, i)

    else:
        for i in iters:
            t = time.time()
            result = algo.train()
            time_acc += time.time()-t
            iters.set_description(f"{i} {args.mode}) Mean Reward = {result['episode_reward_mean']:.2f} | Mean Time = {round(time_acc/(i+1), 3)} | TB: {url} | Progress")

            if i % 200 == 0:
                make_checkpoint(args, algo, log_dir, i, test_env)
