# Entropy-guided Masked Diffusion Model
## Make diffusion model learn to predict high-information token first

First of all I want to thank the creators of the dLLM github repo who helped me shape my idea into something real without having to start from scratch. Huge thank you.

**Also** I am an **undergrad** student in CentraleSupélec, France that is doing the work alone. This is a first version, with maybe some imprecisions. Feel free to correct me or contact me so I can cite your paper and review my statements. 

Contact informations at the bottom of this file. I am currently looking for an internship.

### The intuition behind the model

The idea is simple : predict token that provide high information about the context first and then the other words. For instance, in the sentence "The cat sleeps", it is more important to have "cat" and "sleeps", since it provide the information needed. If I say "cat sleeps", we understand about the same thing.

When I first discovered MDM, I thought that what they were supposed to do. Be more precise because it is able to see everything through attention and thus does not limit itself to predict the next word as in AR models, but rather can oversight few tokens ahead to shape what it is about to say.

I hope we can achieve high precision with less parameters by shaping a more predictive architecture (in the sense of getting few steps ahead compared to AR models)

### How I want to proceed

#### Shannon's Entropy

I understand Shannon's entropy as a mean surprise of each token in a distribution of words.

$$
H(X) = \sum_{i=1}^{n} p(x_i) \log_2 \frac{1}{p(x_i)}
$$

where $p(x_i)$ is the probability of token $x_i$. The surprise of a token is:

$$
\text{Surprise}(x_i) = \log_2 \frac{1}{p(x_i)}
$$

To predict meaningful words, I think it is interesting to predict high surprise words associated with the "absolute" distribution. The theorical one where $p(x_i)$ is the probability of saying the token $x_i$ in the selected language.

Thus, the goal is to make the model understand that words are more frequent than other (because frequency is linked to probability thanks to the Strong Law of Large Numbers).

#### The Diffusion process
During my researches, I understood that diffusion models tend to replicate the diffusion process but with the time t reversed ($dt<0$). To help the model predict meaningful token first, we need to mask them for $t$ near $1$ during training ($t$ is going from $0$ to $1$ and can be interpreted as time during the forward diffusion process).

### The introduction of a new (but familiar) hyperparameter : the temperature T

So we have our intuition. But now, we need to know how we select the next token. Shall we go all greedy and select always the highest token or do we need to keep the choice stochastic ?
I say the debate has already been seen in autoregressive models so let's introduce our favorite function
#### How softmax help cover every scenario possible

1. We get the surprises for each word. 

Well it is not that simple since we do not have theoritical $p(x_i)$. So let's approximate it by rounding it to the frequencies of the token of our training dataset. Let's remember that it introduces bias, but the more diverse and precise it is, the more we are close to the real probability. We could also argue that, this probability is depending on time and space, but let's take the global mean.

2. The probability of selecting token $x_i$ for masking is given by the temperature-scaled softmax over surprises:

$$
p_{\text{mask}}(x_i) = \frac{\exp\left(\frac{S(x_i)}{T}\right)}{\sum_{j=1}^{L} \exp\left(\frac{S(x_j)}{T}\right)}
$$

where:
- $S(x_i) = \log_2 \frac{1}{p(x_i)}$ is the surprise of token $x_i$
- $T > 0$ is the temperature parameter
- $L$ is the sequence length
- The sum is over all valid tokens in the current sequence

The temperature $T$ controls the distribution sharpness:
- $T \to 0$: Greedy selection (always mask highest surprise token)
- $T = 1$: Standard softmax (balanced exploration)
- $T \to \infty$: Uniform selection (all tokens equally likely **Usual MDM/dLLM**)

With that method, we connect a continuum of models. We can retrieve the originals MDM models and go up to a greedy selection. We have to tweak T in order to see where we are going and if this path is relevant.

## Experiments

That continuum should help us select the most relevant diffusion model by chosing the right parameter T.

I want to train small models for differents temperatures. Then, I'll plot the loss after few batches and hope it has a minimum somewhere. After finding one or other minimums I hope I will be able to train a much larger model.

Since I only have access to a T4 for the moment, I will train 10 to 20 M parameters models with 15M - 30M tokens. We are not using Chinchilla scaling law for these experiments since we just need quick updates on the impact of T.

To do so, I take the LLaDA architecture and train it to achieve a 15M - 20M (to precise) guided model with these parameters:
```
d_model = 256
n_layers = 4
n_heads = 4
vocab_size = 50K
```
And this truncation of the dataset :
```
eval_samples = 2000
max_length = 256
batch_size = 16
```

I'll go over multiple Ts and share the plot step by step.

I prefer training from scratch rather than fine-tuning LLaDA or DREAM because these are models with $T \to \infty$ and a lot of parameters and I can't compete with my compute power.
## Installation

*From dLLM repo*
```
# create and activate conda environment
conda create -n dllm python=3.10 -y
conda activate dllm

# install pytorch with CUDA 12.4 (other pytorch/cuda versions should also work)
conda install cuda=12.4 -c nvidia
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# install dllm package
pip install -e .
```

Then go to ``` dllm/train/train_egdllm.py ``` and launch training with the hyperparameters you want.
## Ideas to Explore
If Temperature is a relevant hyperparameter :
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

## Ressources

⚠️ This is an active research project. Results are preliminary and under continuous development. Feedback and contributions are welcome!

If you are interested in the project, please read CONTRIBUTIONS.md

To see the modifications, please read MODIFICATIONS.md

Thanks again to the dLLM Github team
```
@misc{dllm,
    author = {Zhanhui Zhou and Lingjie Chen and Hanghang Tong and Dawn Song},
    title = {dLLM: Simple Diffusion Language Modeling},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ZHZisZZ/dllm}},
}
```
This project inherits the MIT license from the original dLLM repository. See LICENSE for details.

If you use this code or find the ideas helpful, please cite:
```
@misc{egmdm,
    author = {Adrien Leveuf},
    title = {Entropy-Guided Masked Diffusion Model},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/adddddddddddddddd/entropy-guided-masked-diffusion-model}},
}
```

Contact : https://www.linkedin.com/in/adrien-leveuf/
