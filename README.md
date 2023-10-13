# InferringTruth

Can people infer the truth from false messages when they know the sender's goal and cost function for generating larger lies?

## Exp/
"marble-flip-2" on server

Goal conditions = {overestimate, underestimate}; Cost function conditions = {linear, quadratic}
Fixed error from 1st version of experiment -- initial instructions always saying "But you don't gain any points if your opponent guesses 45 or below" regardless of over vs underestimate condition. 

Time delay between trials is also stated to be a server issue, not the participant making a decision.


## simulations/
AI.R compares our experiment AI's betabinomial distribution vs a standard binomial distribution.

ABM.Rmd displays the probabilistic simulations that we feature in the manuscript.
