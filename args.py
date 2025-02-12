import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='training and testing')
    parser.add_argument("--root", type=str, default="/scratch/x2895a03/research/md-diffusion/Ashesh")
    parser.add_argument("--dataset-path", type=str, default="dataset") 
    parser.add_argument("--result-path", type=str, default="results") 
    parser.add_argument("--model-type", type=str, default="egnn")
    parser.add_argument('--temperature', type=int, default=300)
    parser.add_argument("--exp_name", type=str, default="egnn_3lr_3e-4_t250")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=501)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=10)

    parser.add_argument("--timesteps", type=int, default=250)

    return parser

if __name__ == "__main__":
    parser = get_parser() 
    args_dict = parser.parse_args()