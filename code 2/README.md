cd julia

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate data from dynamics
python scripts/generate_data.py --config configs/default.yaml

# Train a baseline model (add an experiment name)
python scripts/train.py --config configs/default.yaml --set train.lr=1e-3 train.epochs=20 --exp baseline-$(date +%Y%m%d-%H%M)

# Plot curves for that run
python scripts/plot_metrics.py --run experiments/<your-run-id>
