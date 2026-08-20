# Plain-Language Explainer: The Stambaugh Bias

## The question

The classic return-predictability regression asks whether a variable like dividend yield can forecast future stock returns:

$$
r_{t+1} = \alpha + \beta x_t + u_{t+1}
$$

Here, $r_{t+1}$ is next period's return, $x_t$ is this period's dividend yield, and $\beta$ measures the true forecasting relationship.



If $\beta > 0$, higher dividend yield predicts higher future returns.  
If $\beta = 0$, dividend yield has no true forecasting power.

## The problem

Dividend yield is not a clean predictor.

Dividend yield is built from price:

$$
\text{Dividend Yield} = \frac{\text{Dividends}}{\text{Price}}
$$

Price also drives returns. So the same price movement affects both the return and the predictor.

If price unexpectedly rises, the current return is high, but dividend yield falls because price is in the denominator. If price unexpectedly falls, the current return is low, but dividend yield rises.

That means the return shock and the dividend-yield shock are mechanically linked.

## Why Stambaugh models the predictor too

Stambaugh does not only model the return regression. He also models the predictor itself as a stochastic variable:

$$
x_{t+1} = \theta + \rho x_t + v_{t+1}
$$

This matters because dividend yield is persistent. It moves slowly over time, so $\rho$ is close to one.

The key ingredients are:

1. dividend yield is persistent;
2. dividend yield is price-based;
3. the return shock $u_{t+1}$ and predictor shock $v_{t+1}$ are negatively correlated.

Together, these facts can push the estimated slope upward in finite samples.

## Beta versus beta-hat

A useful distinction is:

- $\beta$ is the true relationship;
- $\hat{\beta}$ is the estimate we get from a sample.

Even if the true $\beta$ is zero, the estimated $\hat{\beta}$ can still come out positive because of the finite-sample bias.

So a positive slope does not automatically prove real predictability.

## The takeaway

The paper is not saying dividend yield can never predict returns.

The paper is saying that standard OLS can make the evidence look stronger than it really is.

A positive slope has to beat the bias, not just beat zero.


Want to see it happen? The <a href="../playground.html">interactive playground</a>
lets you set the true slope to zero, drag the persistence up, and watch the
estimated slopes pile up on the positive side anyway.
