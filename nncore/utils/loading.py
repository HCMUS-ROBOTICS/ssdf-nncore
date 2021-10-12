def load_yaml(path):
    import yaml
    with open(path, 'rt') as f:
        return yaml.safe_load(f)
