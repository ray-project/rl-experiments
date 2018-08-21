## Benchmarks

We benchmark RLlib against published results.

#### IMPALA

`rllib train -f atari-impala/atari-impala.yaml`

IMPALA on 10M time-steps (**40M frames**). Results compared to learning curves from [Mnih et al, 2016](https://arxiv.org/pdf/1602.01783.pdf) extracted at 10M time-steps from Figure 3.

|env|RLlib IMPALA 32-workers|Mnih et al A3C 40M 16-threads|
|---|---|---|
|BeamRider|2071|~3000|
|Breakout|385|~150|
|QBert|4068|~1000|
|SpaceInvaders|719|~600|

IMPALA vs A3C after 1 hour of training:

|env|RLlib IMPALA 32-workers|Mnih et al A3C 1h 16-threads|
|---|---|---|
|BeamRider|3181|~1000|
|Breakout|538|~10|
|QBert|10850|~500|
|SpaceInvaders|843|~300|

![tensorboard](/atari-impala/atari-impala.png)

#### See also

These benchmarks are a work in progress. For other results to compare against, see [yarlp](https://github.com/btaba/yarlp) and [more plots](https://github.com/openai/baselines-results/blob/master/acktr_ppo_acer_a2c_atari.ipynb) from OpenAI.
