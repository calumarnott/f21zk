import argparse, json, random, yaml
import numpy as np, torch
import importlib

def import_system(system_name: str):
    """ Import a dynamics module by system name.

    Args:
        system_name (str): system name, e.g. "pendulum"

    Raises:
        ModuleNotFoundError: _if the module cannot be found

    Returns:
        module: the imported dynamics module
    """
    # e.g. "pendulum" -> "src.pendulum_ml.dynamics.pendulum"
    # mod = importlib.import_module(f"src.pendulum_ml.dynamics.{system_name}")
    try:
        mod = importlib.import_module(f"pendulum_ml.dynamics.{system_name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Could not find dynamics module for system '{system_name}'. Ensure 'src/pendulum_ml/dynamics/{system_name}.py' exists.") from e
    return mod

def seed_all(seed:int=42):
    """ Set random seed for reproducibility.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cfg(path:str): 
    """ Load configuration from a YAML or JSON file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    text = open(path).read()
    if path.endswith(".yaml") or path.endswith(".yml"):
        return yaml.safe_load(text)
    elif path.endswith(".json"):
        return json.loads(text)
    else:
        raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")

def apply_overrides(cfg:dict, pairs:list[str]|None):
    """ Apply command-line overrides to a configuration dictionary.

    Args:
        cfg (dict): Base configuration dictionary.
        pairs (list[str] | None): List of key-value pairs in the format "key=value" to override.

    Returns:
        dict: Updated configuration dictionary with overrides applied.
    """
    
    for kv in (pairs or []): # if pairs is None, do nothing (empty list)
        
        k, v = kv.split("=", 1) # split only on the first '='
        
        d = cfg 
        
        *ks, last = k.split(".") # e.g. train.lr -> ks=["train"], last="lr"
        
        for kk in ks: 
            # iterate through intermediate keys, creating nested dicts as needed 
            # E.g. if "train" not in cfg, set cfg["train"] = {}. if "train" in cfg, d = cfg["train"]
            d = d.setdefault(kk, {}) 
            
        val = v
        if v.lower() in {"true","false"}: val = (v.lower()=="true") # if v is "true" or "false", convert to bool
        else:
            try: val = int(v) # try to convert to int
            except:
                try: val = float(v) # try to convert to float
                except: pass
                
        d[last] = val # set the final key to the value
        
    return cfg

def parse_with_config():
    """ Parse command-line arguments and load configuration.
    e.g. python train.py --config configs/pendulum.yaml --set train.lr=3e-4 train.epochs=50 --exp my_experiment

    Returns:
        tuple: Configuration dictionary and parsed arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*")           # e.g. train.lr=3e-4 train.epochs=50
    p.add_argument("--exp", default=None)
    
    args = p.parse_args() # parse command-line arguments
    
    # apply command-line overrides
    cfg = apply_overrides(load_cfg(args.config), args.set) 
    
    if args.exp: cfg["exp"] = args.exp # set experiment name if provided
    
    # set in_dim and out_dim based on system
    cps = import_system(cfg["system"])
    
    name = cfg["model"]["name"]
    cfg["model"][name]["in_dim"] = len(cps.STATE_NAMES) + len(cps.INPUT_ERROR_AXES) # state + error for each control axis
    cfg["model"][name]["out_dim"] = len(cps.OUTPUT_CONTROL_AXES) # control for each control axis
    
    seed_all(cfg.get("seed", 0))
    
    return cfg, args
