## RLlib Atari Results

Benchmarks of [RLlib](https://rllib.io) algorithms against published results. These benchmarks are a work in progress. For other results to compare against, see [yarlp](https://github.com/btaba/yarlp) and [more plots](https://github.com/openai/baselines-results/blob/master/acktr_ppo_acer_a2c_atari.ipynb) from OpenAI.

#### IMPALA and A2C

`rllib train -f atari-impala/atari-impala.yaml`

`rllib train -f atari-a2c/atari-a2c.yaml`

RLlib IMPALA and A2C on 10M time-steps (**40M frames**). Results compared to learning curves from [Mnih et al, 2016](https://arxiv.org/pdf/1602.01783.pdf) extracted at 10M time-steps from Figure 3.

|env|RLlib IMPALA 32-workers|RLlib A2C 5-workers|Mnih et al A3C 40M 16-threads|
|---|---|---|---|
|BeamRider|2071|1401|~3000|
|Breakout|385|374|~150|
|QBert|4068|3620|~1000|
|SpaceInvaders|719|692|~600|

IMPALA and A2C vs A3C after 1 hour of training:

|env|RLlib IMPALA 32-workers|RLlib A2C 5-workers|Mnih et al A3C 1h 16-threads|
|---|---|---|---|
|BeamRider|3181|874|~1000|
|Breakout|538|268|~10|
|QBert|10850|1212|~500|
|SpaceInvaders|843|518|~300|

IMPALA plots:
![tensorboard](/atari-impala/atari-impala.png)

A2C plots:
![tensorboard](/atari-a2c/atari-a2c.png)

#### DQN

`rllib train -f atari-dqn/basic-dqn.yaml`

RLlib DQN after 10M time-steps (**40M frames**).

| env  |  RLlib Basic DQN |  Prio. Duel DDQN | Rainbow  |  Hessel et al. DQN |  Hessel et al. Rainbow |
|---|---|---|---|---|---|
|BeamRider|2869|TODO|TODO|~2000|~13000|
|Breakout|287|TODO|TODO|~150|~300|
|QBert|3921|TODO|TODO|~4000|~20000|
|SpaceInvaders|650|TODO|TODO|~500|~2000|

Basic DQN plots:
![tensorboard](/atari-dqn/basic-dqn.png)

#### PPO

`rllib train -f atari-ppo/atari-ppo.yaml`

RLlib PPO after 10M time-steps (**40M frames**). TODO: these results can probably be improved with more tuning.

|env|RLlib PPO 10-workers|
|---|---|
|BeamRider|1206|
|Breakout|122|
|QBert|9847|
|SpaceInvaders|682|

PPO plots:
![tensorboard](/atari-ppo/atari-ppo.png)

#### DQN

`rllib train -f atari-dqn/basic-dqn.yaml`

`rllib train -f atari-dqn/dueling-ddqn.yaml`

`rllib train -f atari-dqn/rainbow.yaml`

RLlib DQNs after 10M time-steps (**40M frames**).

|env|Basic DQN|Dueling DDQN|Rainbow|Hessel et a. DQN|Hessel et al. Rainbow|
|---|---|---|---|---|
|BeamRider|2724|1074|2200|~2000|~13000|
|Breakout|281|38|72|~150|~300|
|QBert|796|3517|1059|~4000|~20000|
|SpaceInvaders|629|459|700|~500|~2000|
