import argparse
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from nncore.utils.loading import load_yaml


class Opts(Namespace):

    @staticmethod
    def _argparse(parent: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
        if parent is None:
            parser = argparse.ArgumentParser()
        else:
            parser = parent.add_argument_group()

        parser.add_argument("--id")
        parser.add_argument("--test", action="store_true", default=False)
        parser.add_argument(
            "--debug",
            type=int,
            help="level of visualization."
            "1: only show the final detection results"
            "2: show the network output features"
            "3: use matplot to display"  # useful when lunching training with ipython notebook
            "4: save all visualizations to disk",
        )
        parser.add_argument(
            "--demo", help="save lastest sample", action="store_true", default=False
        )
        parser.add_argument(
            "--fp16",
            help="use floating point (only work in gpu machine)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--load-model", default="", help="path to pretrained model"
        )
        parser.add_argument(
            "--cfg-pipeline", default=None, help="path to pipeline yaml"
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="resume an experiment. "
            "Reloaded the optimizer parameter and "
            "set load_model to model_last.pth "
            "in the exp dir if load_model is empty.",
        )

        # system
        parser.add_argument(
            "--gpus", help="-1 for CPU, use comma for multiple gpus", nargs='*'
        )
        parser.add_argument(
            "--num-workers", type=int, help="dataloader threads. 0 for single-thread.",
        )
        parser.add_argument("--seed", type=int, help="random seed")
        # log
        parser.add_argument(
            "--verbose", type=int, help="disable progress bar and print to screen.",
        )
        parser.add_argument("--config-path", type=str)

        # train
        parser.add_argument("--nepochs", type=int, help="total training epochs.")
        parser.add_argument("--batch-size", type=int, help="batch size")
        parser.add_argument(
            "--num-iters", type=int, help="default: #samples / batch_size."
        )
        parser.add_argument(
            "--val-step", type=int, help="number of epochs to run validation.",
        )

        parser.add_argument(
            "--log-step", type=int, help="number of epochs to logging.",
        )
        parser.add_argument(
            "--save-dir", type=str, help="saving path",
        )

        if parent is None:
            return parser

        return parent

    @staticmethod
    def _fill(a: Dict, b: Dict) -> Dict:
        for k, v in b.items():
            if a.get(k, None) is None:
                a.update({k: v})
        return a

    @staticmethod
    def parse(cfg_path: str, use_argparse: Union[str, ArgumentParser] = 'none') -> 'Opts':
        cfg = load_yaml(cfg_path)['opts']

        if use_argparse == 'none':
            args = {}
        elif use_argparse == 'default':
            args = vars(Opts._argparse(None).parse_args())
        elif isinstance(use_argparse, ArgumentParser):
            args = vars(use_argparse.parse_args())
        else:
            raise ValueError('Unsupported use_argparse value. It must be'
                             '["none", "default" or an ArgumentParser instance]')

        args = Opts._fill(args, cfg)
        opts = Opts(**args)

        # no flag -> cpu
        # --gpus -> use all
        # --gpus [index] 0, 1, 2 -> use 0,1,2 (3 gpus)
        # --gpus
        if args.get('gpus', None) is None or not torch.cuda.is_available():
            opts.device = 'cpu'
        elif isinstance(args['gpus'], int):
            if args['gpus'] == -1:
                opts.device = 'cuda'
            elif args['gpus'] >= 0:
                opts.device = f'cuda:{args["gpus"]}'
            else:
                raise ValueError('Invalid GPU index')
        elif isinstance(args['gpus'], list):
            opts.device = f'cuda:{",".join(map(str, args["gpus"]))}'
        elif isinstance(args["gpus"], str):
            opts.device = f'cuda:{args["gpus"]}'

        output_name = f'{opts.id}_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}'
        opts.save_dir = Path(opts.save_dir) / output_name
        print(f"The output will be saved to {opts.save_dir}")
        return opts
