import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    
    args = parser.parse_args()

    return args

def main():
    print("start")
    args = parse_args()
    print(args)

    # path = 'output/yolof_2048_adapt/'

    for file in os.listdir(args.path):
        if file.endswith(".pth"):
            whole_path = os.path.join(args.path, file)
            print(whole_path)
            checkpoint = torch.load(whole_path)
            checkpoint['meta']['config'] = '#' + checkpoint['meta']['config']
            torch.save(checkpoint, whole_path)

if __name__ == '__main__':
    main()