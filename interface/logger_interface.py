from logger import *
import os.path as osp

def build_logger(log_dir, args, configs):
    logger_name = configs.logger.name
    logger = globals()[f'{logger_name}'](
        args,
        configs,
        log_file=log_dir,
        log_interval=args.log_interval,
    )
    return logger