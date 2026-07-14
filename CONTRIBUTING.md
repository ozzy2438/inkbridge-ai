# Contributing to InkBridge AI

## Development Setup

```bash
# Clone and setup
git clone https://github.com/ozzy2438/inkbridge-ai.git
cd inkbridge-ai
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
make test

# Run linting
make lint

# Start development server
make dev
```

## Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `experiment/*`: ML experiments (may not merge)

## Pull Request Process

1. Create feature branch from `develop`
2. Write tests for new functionality
3. Ensure all tests pass: `make test`
4. Ensure linting passes: `make lint`
5. Update documentation if needed
6. Submit PR with clear description

## Code Style

- Python 3.10+
- Formatted with Black (line length 100)
- Linted with Ruff
- Type hints required for public APIs
- Docstrings for all public functions

## ML Experiment Guidelines

- Track all experiments with clear naming
- Log hyperparameters and results
- Never commit model weights to git
- Use writer-independent splits
- Report all metrics (not just best)
