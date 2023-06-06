import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    
def convert_to_int(config):
    config = [int(i) for i in config.values()]
    return config