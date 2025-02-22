import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='training and testing')
    parser.add_argument("--root", type=str, default="/scratch/x2895a03/research/md-diffusion/Ashesh")
    parser.add_argument("--dataset-path", type=str, default="dataset") 
    parser.add_argument("--result-path", type=str, default="results") 
    parser.add_argument("--model-type", type=str, default="egnn")

    # dataset args
    parser.add_argument('--temperature', type=int, nargs='+', default=[1000])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_bunch", type=int, default=1)
    parser.add_argument("--n_offset", type=int, default=1)
    parser.add_argument("--do_norm", type=bool, default=False)
    #parser.add_argument('--temperature', type=int, default=1000)
    parser.add_argument("--t_selection", type=int, nargs='+', default=[1000])# [200, 300, 500] / 1000
    parser.add_argument("--t_to_simulate", type=int, default=1000)

    # experiment args
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=2501)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=50)

    # how to do inference?
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--how_to_sample", type=str, default="one_step", choices=["one_step", "one_step_diff", "next_frame", "direct"])

    return parser

if __name__ == "__main__":
    parser = get_parser() 
    args_dict = parser.parse_args()