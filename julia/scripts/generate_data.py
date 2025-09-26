from pendulum_ml.utils import parse_with_config
from pendulum_ml.data.generate import simulate
from pendulum_ml.data.process import to_processed

if __name__ == "__main__":
    """ Generate and process data based on configuration. """
    
    cfg, _ = parse_with_config() # get config from command-line args
    
    # simulate and save raw data
    raw_paths = simulate(cfg, out_dir=f"data/raw/{cfg['system']}") 
    
    # process raw data and save processed tensors
    to_processed(raw_paths, cfg, out_dir=f"data/processed/{cfg['system']}")
