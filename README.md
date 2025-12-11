# Entropy-guided Masked Diffusion Model
## Make diffusion model learn to predict high-information token first

First of all I want to thank the creators of the dLLM github repo who helped me shape my idea into something real without having to start from scratch. Huge thank you.

Also I am an undergrad student in CentraleSupélec, France that is doing the work alone. This is a first version, with maybe some imprecisions. Feel free to correct me or contact me so I can cite your paper and review my statements.

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
- $T \to \infty$: Uniform selection (all tokens equally likely even if two tokens appear in the same sequence since they are different by their position, another variable we want the model to understand. e. g. we want the model to understand where the meaningful tokens lay.)

With that method, we connect a continuum of models.
