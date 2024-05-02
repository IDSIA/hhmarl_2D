from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog
import numpy as np
import torch
import os
import tqdm
from config import Config
import time
import json
from pathlib import Path
from envs.env_hier import HighLevelEnv
from models.ac_models_hier import CommanderGru
ModelCatalog.register_custom_model("commander_model",CommanderGru)

N_OPP_HL = 2 #sensing
OBS_DIM = 14+10*N_OPP_HL

MODEL_NAME = "Commander_3_vs_3" #name of commander model in folder 'results'
N_EVALS = 1000


def cc_obs(obs):
    return {
        "obs_1_own": obs,
        "obs_2": np.zeros(OBS_DIM, dtype=np.float32),
        "obs_3": np.zeros(OBS_DIM, dtype=np.float32),
        "act_1_own": np.zeros(1),
        "act_2": np.zeros(1),
        "act_3": np.zeros(1),
    }

def evaluate(args, env, algo, epoch, eval_stats, eval_log):
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    info = {}

    while not done:
        actions = {}
        states = [torch.zeros(200), torch.zeros(200)]

        if args.eval_hl:
            for ag_id, ag_s in state.items():
                a = algo.compute_single_action(obs=cc_obs(ag_s), state=states, explore=False)
                actions[ag_id] = a[0]
                states[0] = a[1][0]
                states[1] = a[1][1]
        else:
            # if no commander involved, assign closest opponent for each agent. 
            for n in range(1,args.num_agents+1):
                actions[n] = 1
        state, rew, hist, trunc, info = env.step(actions)
        for r in rew.values(): reward += r
        done = hist["__all__"] or trunc["__all__"]
        step += 1

        for k, v in info.items(): eval_stats[k] += v
        eval_stats["total_n_actions"]+=1

    if epoch %100 ==0:
        env.plot(Path(eval_log, f"Ep_{epoch}_Step_{step}_Rew_{round(reward,2)}.png"))

    return

def postprocess_eval(ev, eval_file):
    #calculate fractions
    win = (ev["agents_win"] / N_EVALS) * 100
    lose = (ev["opps_win"] / N_EVALS) * 100
    draw = (ev["draw"] / N_EVALS) * 100
    fight = (ev["agent_fight"] / ev["agent_steps"]) *100
    esc = (ev["agent_escape"] / ev["agent_steps"]) *100
    fight_opp = (ev["opp_fight"] / ev["opp_steps"]) *100
    esc_opp = (ev["opp_escape"] / ev["opp_steps"]) *100
    opp1 = (ev["opp1"] / ev["agent_fight"]) *100
    opp2 = (ev["opp2"] / ev["agent_fight"]) *100
    opp3 = (ev["opp3"] / ev["agent_fight"]) *100
    evals = {"win":win, "lose":lose, "draw":draw, "fight":fight, "esc":esc, "fight_opp":fight_opp, "esc_opp":esc_opp, "opp1":opp1, "opp2":opp2, "opp3":opp3}
    for k,v in evals.items():
        print(f"{k}: {round(v,2)}")
    with open(eval_file, 'w') as file:
        json.dump(evals, file, indent=3)

if __name__ == "__main__":
    t1 = time.time()
    args = Config(2).get_arguments

    log_base = os.path.join(os.getcwd(),'results')
    check = os.path.join(log_base, MODEL_NAME, 'checkpoint')
    config = "Commander_" if args.eval_hl else "Low-Level_"
    config = config + f"{args.num_agents}-vs-{args.num_opps}"
    eval_log = os.path.join(log_base, "EVAL_"+config)
    eval_file = os.path.join(eval_log, f"Metrics_{config}.json")
    if not os.path.exists(eval_log): os.makedirs(eval_log)
    
    env = HighLevelEnv(args.env_config)

    # if evaluating purely low-level policies, we don't need commander.
    policy = Policy.from_checkpoint(check, ["commander_policy"])["commander_policy"] if args.eval_hl else None
    eval_stats = {"agents_win": 0, "opps_win": 0, "draw": 0, "agent_fight": 0, "agent_escape":0, "opp_fight":0, "opp_escape":0, \
                  "agent_steps":0, "opp_steps":0, "total_n_actions":0 ,\
                    "opp1":0, "opp2":0, "opp3":0}
    iters = tqdm.trange(0, N_EVALS,  leave=True)
    for n in iters:
        evaluate(args, env, policy, n, eval_stats, eval_log)

    print("------RESULTS:")
    postprocess_eval(eval_stats, eval_file)
    print(f"------TIME: {round(time.time()-t1, 3)} sec.")
