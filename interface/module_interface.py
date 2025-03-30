from module import *

def build_module(args, configs):
    model_name = configs.model.name
    model = globals()[f'{model_name}'](configs)
    return model