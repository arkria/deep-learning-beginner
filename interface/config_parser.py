import os
import os.path as osp
import yaml
from easydict import EasyDict as edict


def parse_config(config_file, args):
    config = edict(load_config(config_file))
    assert 'task_name' in config
    args_vars = vars(args)
    for k, v in args_vars.items():
        if k.startswith('@') and k.split('@')[-1].split('.')[0] == config['task_name']:
            modify_config(config, k, v)
    return config


def load_config(config_file):
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f)
    return configs

def modify_config(config, args_name, args_value):
    args_path = args_name.split('.')
    assert args_path[1] in config
    args_dict = create_nested_dict(args_path[1:], args_value)
    update_config(config, args_dict)


def create_nested_dict(keys, final_value):
    if not keys:  # 如果键列表为空，直接返回最终值
        return final_value
    
    # 从最后一个键开始递归构建嵌套字典
    nested_dict = final_value
    for key in reversed(keys):
        nested_dict = {key: nested_dict}
    
    return nested_dict


def update_config(config, args_dict):
    """
    递归更新嵌套字典 config，根据 args_dict 的内容。
    如果 args_dict 中的键在 config 中存在，则更新其值；
    如果不存在，则在 config 中添加新的键值对。
    """
    for key, value in args_dict.items():
        # 如果 value 是字典，递归更新
        if isinstance(value, dict):
            if key in config and isinstance(config[key], dict):
                update_config(config[key], value)
            else:
                config[key] = value  # 如果 config 中不存在该键，直接添加
        else:
            config[key] = value