cd implementation

<!-- python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pip install -e .  -->

# Start a conda environment
conda env create -f environment.yml
conda activate pendulum-ml



# Generate data from dynamics
python scripts/generate_data.py 

# Train a baseline model (add an experiment name)
python scripts/train.py --config configs/default.yaml --set train.lr=1e-3 train.epochs=20 --exp baseline-$(date +%Y%m%d-%H%M)

# Plot curves for that run
python scripts/plot_metrics.py --run experiments/<your-run-id>

# Evaluate the model
python scripts/evaluate.py --run <your-run-id>

# Add new Cyber Physical System (CPS)
1. Add a new dynamics file in pendulum_ml/dynamics
2. Add a new config file in configs for the new CPS
3. In the dynamics file, each module (pendulum.py, quadcopter.py) should have:
   - A function `f(state, control, params)` that returns the system derivatives, given the current state, control input, and system parameters. This represents the next state of the system.
   - A `Params` @dataclass class that specifies the names of the parameter keys the system expects under `dynamics` in the config file.
   - `CONTROL_AXES`: the control channels it supports (e.g., ['theta'] for pendulum, ['x', 'z', 'theta'] for quadcopter).
   - `error(axis, x, setpoint)`: a function that computes the error for a given axis.




# HOW TO TODOs:
- How to generate data for different CPSs?
- How to train a model
- Where data is stored
- How to evaluate that model
- How to add a new CPS
- which configurations can you override and how to do it
- All is run from the scripts
- How to run adversarial attack
- How to visualize results and create animations for different CPSs
- Full pipeline example
# TODOs:
- Data process file. Automate model input/output dimension detection from data
- Add quadcopter dynamics file
- Add quadcopter config file
- Extend controller to support inner-loop controllers (e.g., for quadcopter, have a position controller that outputs desired angles, which are then fed to an altitude controller)
- Add animation scripts for pendulum and quadcopter
- 
