import torch
import argparse
import gymnasium as gym
from models import *
from utils import *
import env_rl_othello


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_episode", type=int, default=1)
    parser.add_argument("--recording", type=bool, default=False)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rssm_continue", action="store_true")
    parser.add_argument("--state_size", type=int, default=512) # 1024
    parser.add_argument("--categorical_size", type=int, default=32) # 32
    parser.add_argument("--class_size", type=int, default=16) # 32
    parser.add_argument("--deterministic_size", type=int, default=256) # 512
    parser.add_argument("--hidden_size", type=int, default=128) # 512
    parser.add_argument("--observation_size", type=int, default=128) # 512
    parser.add_argument("--discrete_action", type=bool, default=True)
    parser.add_argument("--continuous_action", type=bool, default=True)
    parser.add_argument("--mean_scale", type=float, default=5.0)
    parser.add_argument("--min_std", type=float, default=1e-4)
    args = parser.parse_args()

    return args


@torch.no_grad()
def main():
    args = parse_args()
#    env = gym.make("CarRacing-v2", render_mode="rgb_array")
#    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = env_rl_othello.DreamerOthelloEnv()
#    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.rssm_continue:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    obs_shape = env.observation_space.shape
#    obs_shape = obs_shape[-1:] + obs_shape[:2]

    encoder = Encoder2D(args, obs_shape[0]).to(device)
    recurrent = RSSM(args, action_dim).to(device)
    representation = RepresentationModel(args).to(device)
    if args.rssm_continue:
        actor = ActionContinuous(args, action_dim).to(device)
    else:
        actor = ActionDiscrete(args, action_dim).to(device)

    encoder.load_state_dict(torch.load(args.output + "_trained/encoder.pth"))
    recurrent.load_state_dict(torch.load(args.output + "_trained/recurrent.pth"))
    representation.load_state_dict(torch.load(args.output + "_trained/representation.pth"))
    actor.load_state_dict(torch.load(args.output + "_trained/actor.pth"))

    if args.recording:
        env = gym.wrappers.RecordVideo(env, video_folder=args.output,
                                       episode_trigger=lambda x: x % 1 == 0)
        assert args.total_episode == 1
    total_reward = 0
    for episode in range(args.total_episode):
        obs, info = env.reset()
#        obs = normalize_obs(obs)
        done = False
        deter = recurrent.init_hidden(1).to(device)
        state = recurrent.init_state(1).to(device)
        action = torch.zeros(1, action_dim).to(device)
        while not done:
            obs_embed = encoder(torch.tensor(
                obs, dtype=torch.float32).to(device).unsqueeze(0))
            deter = recurrent(state, action, deter)
            _, state = representation(deter, obs_embed)
            _, action = actor(state, deter, training=False)

            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
#            next_obs = normalize_obs(next_obs)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
            print("reward: ", reward)
            print("board:\n", env.board[0] + env.board[1]*2 + env.board[4]*10)
    print("total reward:", total_reward)
#    env.close()


if __name__ == "__main__":
    main()
