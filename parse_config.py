import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json

import data_loader.datasets as Datasets
import data_loader.transforms as Transforms


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.run_id = run_id
        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        if self.run_id is None:  # use timestamp as default run-id
            self.run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = save_dir / "models" / exper_name / self.run_id
        self._log_dir = save_dir / "log" / exper_name / self.run_id

        # make directory for saving checkpoints and log.
        exist_ok = self.run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / "config.json")

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args, options=""):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        # if args.device is not None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        os.environ["CUDA_VISIBLE_DEVICES"] = "cuda"

        # if args.resume is not None:
        #     resume = Path(args.resume)
        #     cfg_fname = resume.parent / "config.json"
        #     if args.config:
        #         cfg_fname = Path(args.config)
        # else:
        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        # assert args.config is not None, msg_no_cfg
        resume = None
        # cfg_fname = Path(args.config)

        # config = read_json(cfg_fname)

        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(resume, modification)
        # return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"

        # DataLoader??? ?????? ??????
        if "data_loader" in name:
            module_args["dataset"] = self._get_dataset(module_args, *args, **kwargs)

        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def _get_dataset(self, module_args, *args, **kwargs):
        arg_name = module_args["dataset"]["type"]
        if arg_name == "ConcatDataset":
            arg_args = {"datasets": []}
            for sub_args in module_args["dataset"]["args"]:
                arg_args["datasets"].append(self._get_dataset(sub_args, *args, **kwargs))
        else:
            arg_args = dict(module_args["dataset"]["args"])
            if "transform" in arg_args:
                arg_args["transform"] = self._get_transform(arg_args, *args, **kwargs)
        return getattr(Datasets, arg_name)(*args, **arg_args)

    def _get_transform(self, module_args, *args, **kwargs):
        arg_name = module_args["transform"]["type"]
        arg_args = dict(module_args["transform"]["args"])
        return getattr(Transforms, arg_name)(*args, **arg_args)

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):  # setter
        self._save_dir = value
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def get_run_id(self):
        return self.run_id


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


def change_fold(config, fold, save_dir=None):
    """?????? fold??? ?????? annotation file ?????? ??????"""
    # kfold??? ???????????? ?????? ?????? ?????? ??????
    assert config["kfold"]["flag"] is True
    if save_dir is not None:
        config.save_dir = Path(os.path.join(save_dir, f"fold{fold}"))

    ann_train = config["kfold"]["train_fold"][:-5] + str(fold) + config["kfold"]["train_fold"][-5:]
    ann_valid = config["kfold"]["valid_fold"][:-5] + str(fold) + config["kfold"]["valid_fold"][-5:]

    modification = {
        "data_loader;args;dataset;args;ann_file": ann_train,
        "valid_data_loader;args;dataset;args;ann_file": ann_valid,
    }
    return _update_config(config, modification)
