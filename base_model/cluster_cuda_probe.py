import argparse
import json
import socket
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description='Diagnostica o runtime PyTorch/CUDA do ambiente atual.')
    parser.add_argument('--require-cuda', action='store_true', help='Sai com erro se CUDA nao estiver realmente disponivel.')
    args = parser.parse_args()

    info = {
        'hostname': socket.gethostname(),
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count(),
        'devices': [],
    }

    if info['cuda_available']:
        for idx in range(info['cuda_device_count']):
            props = torch.cuda.get_device_properties(idx)
            info['devices'].append(
                {
                    'index': idx,
                    'name': torch.cuda.get_device_name(idx),
                    'total_memory_gb': round(props.total_memory / (1024 ** 3), 2),
                    'major': props.major,
                    'minor': props.minor,
                }
            )

    print(json.dumps(info, indent=2))

    if args.require_cuda and not info['cuda_available']:
        if info['torch_cuda_version'] is None:
            print('STATUS: cpu_only_pytorch_build', file=sys.stderr)
        else:
            print('STATUS: cuda_build_but_not_available', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
