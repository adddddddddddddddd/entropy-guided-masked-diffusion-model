# Contributing to Temperature-Modulated dLLM

We welcome contributions! Here's how you can help:

## Areas for Contribution
- Find the best Temperature setting
- Maybe change the scheduler to train closely to $t=1$ because it needs more training to predict high surprise words.
- Train a model with more parameters
- Run evaluation benchmarks
- Use that method to link autoregressive models with MDM through a different distribution such as 

$$
p_{\text{mask}}(x_i) = \frac{\exp\left(\frac{Toke n\ position}{T}\right)}{\sum_{j=1}^{L} \exp\left(\frac{Toke n\ position}{T}\right)}
$$

It will go from AR to MDM and we can eventually evaluate new hybrid models.

Also, it may exists other paths.


## Getting Started
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## Code Style
- Follow PEP 8
- Add docstrings to functions
- Include type hints where appropriate