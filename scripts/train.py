import sys
import argparse
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning import loggers

sys.path.append('../')
from src.pl_module import ClassifierModule
from src.utils import load_yaml, dump_yaml, get_cur_time_str


def parse_args():
    parser = argparse.ArgumentParser(description='Run Tacotron experiment')
    parser.add_argument(
        '--config', type=Path, required=True,
        help='Path to the config yaml file'
    )
    parser.add_argument(
        '--experiments_dir', type=Path, required=True,
        help='Root directory of all your experiments'
    )
    parser.add_argument(
        '--experiment_name', type=str, required=False, default=None,
        help='Name of current experiment'
    )
    args = parser.parse_args()
    return args


def prepare_experiment(args) -> Path:

    experiments_dir: Path = args.experiments_dir
    experiments_dir.mkdir(exist_ok=True, parents=True)

    if args.experiment_name is None:
        experiment_name = get_cur_time_str()
    else:
        experiment_name: str = args.experiment_name
        if (experiments_dir / experiment_name).is_dir():
            experiment_name = experiment_name + '_' + get_cur_time_str()

    experiment_dir: Path = experiments_dir / experiment_name
    experiment_dir.mkdir(exist_ok=False)

    models_dir = experiment_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    return experiment_dir


def get_trainer(_args: argparse.Namespace, _hparams: Dict) -> pl.Trainer:

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=_hparams['models_dir'],
        save_top_k=1,
        verbose=True
    )

    tb_logger_callback = loggers.TensorBoardLogger(
        save_dir=_hparams['tb_logdir'],
        name='logs'
    )

    trainer_args = {
        'logger': tb_logger_callback,
        'checkpoint_callback': model_checkpoint_callback,
        'max_epochs': _hparams['epochs'],
        'gpus': _hparams['gpus'],
        'gradient_clip_val': _hparams['grad_clip_thresh'],
        'accumulate_grad_batches': _hparams['accum_steps'],
        'show_progress_bar': True,
        'progress_bar_refresh_rate': 1,
        'log_save_interval': 1
    }
    _trainer = pl.Trainer(**trainer_args)
    return _trainer


def main():
    args = parse_args()
    experiment_dir = prepare_experiment(args)

    hparams = load_yaml(args.config)
    dump_yaml(hparams, experiment_dir / 'hparams.yaml')
    hparams['models_dir'] = experiment_dir / 'models'
    hparams['tb_logdir'] = experiment_dir

    module = ClassifierModule(hparams)
    trainer = get_trainer(_args=args, _hparams=hparams)
    trainer.fit(module)


if __name__ == '__main__':
    main()
