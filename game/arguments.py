import argparse
import torch

def get_game_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--participant", type=str, default="test")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=0.0003)
    parser.add_argument("--critic-lr", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=True)
    parser.add_argument("--alpha-lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=[32, 32])
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[32, 32])
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--num-actions", type=int, default=3)
    parser.add_argument("--agent-type", type=str, default="basesac")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--avg-q", action="store_true", default=True)
    parser.add_argument('--clip-q', action="store_true", default=True)
    parser.add_argument("--clip-q-epsilon", type=float, default=0.5)
    parser.add_argument("--entropy-penalty", action="store_true", default=True)

    parser.add_argument('--entropy-penalty-beta',type=float,default=0.5)


    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--update-target-every', type=int, default=1)
    parser.add_argument('--load-model', type=str, default='rl_models/Initial/')

    parser.add_argument('--ere', action='store_true', default=False)
    parser.add_argument('--eta', type=float, default=0.997)
    parser.add_argument('--cmin', type=int, default=2000)

    parser.add_argument('--dqfd',action='store_true',default=False)
    parser.add_argument('--demo-path',type=str,default='game/Demonstration_Buffer/buffer.npy')

    parser.add_argument('--leb',action='store_true',default=False)
    parser.add_argument('--buffer-path',type=str,default='game/Saved_Buffers/')

    parser.add_argument('--ppr',action='store_true',default=False)
    parser.add_argument('--expert-policy',type=str,default='rl_models/Policy_Transfer/')




    return parser.parse_args()

