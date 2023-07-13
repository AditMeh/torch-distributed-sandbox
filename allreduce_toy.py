# https://github.com/narumiruna/pytorch-distributed-example/blob/master/toy/main.py

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import argparse
from random import randint

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def run(world_size, rank, steps):
    for step in range(1, steps + 1):
        # get random int
        value = randint(0, 10)

        # group all ranks
        ranks = list(range(world_size))
        group = dist.new_group(ranks=ranks)
        
        # compute reduced sum
        tensor = torch.tensor(value, dtype=torch.int).to(f'cuda:{rank}')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        
        dist.barrier()

        if rank == 0:
            print('rank: {}, step: {}, value: {}, reduced sum: {}.'.format(rank, step, value, tensor.item()))
        if rank == 1:
            print('rank: {}, step: {}, value: {}, reduced sum: {}.'.format(rank, step, value, tensor.item()))


def setup(rank, world_size, backend= "nccl"):
    if rank != -1:  # -1 rank indicates serial code
        # Initializes the default distributed process group, and this will also initialize the distributed package.
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        print(f'--> done setting up rank={rank}')
        run(world_size, rank, 10)


def main():
    master_addr = '127.0.0.1'
    master_port = find_free_port()
    


    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='nccl', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world_size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()

    # run(args.world_size, args.rank, args.steps)
    mp.spawn(setup, args=[args.world_size], nprocs = args.world_size)
    # Runs setup(i, *args), for i in {0, .. nprocs-} 
    
if __name__ == '__main__':
    main()