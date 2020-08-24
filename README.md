## RLlib Reference Results

Benchmarks of [RLlib](https://rllib.io) algorithms against published results. These benchmarks are a work in progress. For other results to compare against, see [yarlp](https://github.com/btaba/yarlp) and [more plots](https://github.com/openai/baselines-results/blob/master/acktr_ppo_acer_a2c_atari.ipynb) from OpenAI.

#### Ape-X Distributed Prioritized Experience Replay

`rllib train -f atari-apex/atari-apex.yaml`

Comparison of RLlib Ape-X to Async DQN after 10M time-steps (**40M frames**). Results compared to learning curves from [Mnih et al, 2016](https://arxiv.org/pdf/1602.01783.pdf) extracted at 10M time-steps from Figure 3.

|env|RLlib Ape-X 8-workers|Mnih et al Async DQN 16-workers|Mnih et al DQN 1-worker|
|---|---|---|---|
|BeamRider|6134|~6000|~3000|
|Breakout|123|~50|~10|
|QBert|15302|~1200|~500|
|SpaceInvaders|686|~600|~500|

Here we use only eight workers per environment in order to run all experiments concurrently on a single g3.16xl machine. Further speedups may be obtained by using more workers. Comparing wall-time performance after 1 hour of training:

|env|RLlib Ape-X 8-workers|Mnih et al Async DQN 16-workers|Mnih et al DQN 1-worker|
|---|---|---|---|
|BeamRider|4873|~1000|~300|
|Breakout|77|~10|~1|
|QBert|4083|~500|~150|
|SpaceInvaders|646|~300|~160|

Ape-X plots:
![apex](/atari-apex/apex.png)

#### IMPALA and A2C

`rllib train -f atari-impala/atari-impala.yaml`

`rllib train -f atari-a2c/atari-a2c.yaml`

RLlib IMPALA and A2C on 10M time-steps (**40M frames**). Results compared to learning curves from [Mnih et al, 2016](https://arxiv.org/pdf/1602.01783.pdf) extracted at 10M time-steps from Figure 3.

|env|RLlib IMPALA 32-workers|RLlib A2C 5-workers|Mnih et al A3C 16-workers|
|---|---|---|---|
|BeamRider|2071|1401|~3000|
|Breakout|385|374|~150|
|QBert|4068|3620|~1000|
|SpaceInvaders|719|692|~600|

IMPALA and A2C vs A3C after 1 hour of training:

|env|RLlib IMPALA 32-workers|RLlib A2C 5-workers|Mnih et al A3C 16-workers|
|---|---|---|---|
|BeamRider|3181|874|~1000|
|Breakout|538|268|~10|
|QBert|10850|1212|~500|
|SpaceInvaders|843|518|~300|

IMPALA plots:
![tensorboard](/atari-impala/atari-impala.png)

A2C plots:
![tensorboard](/atari-a2c/atari-a2c.png)

#### Pong in 3 minutes
With a bit of tuning, RLlib IMPALA can solve Pong in ~3 minutes:

`rllib train -f pong-speedrun/pong-impala-fast.yaml`

![tensorboard](/pong-speedrun/pong-impala.png)

#### DQN / Rainbow

`rllib train -f atari-dqn/basic-dqn.yaml`
`rllib train -f atari-dqn/duel-ddqn.yaml`
`rllib train -f atari-dqn/dist-dqn.yaml`

RLlib DQN after 10M time-steps (**40M frames**). Note that RLlib evaluation scores include the 1% random actions of epsilon-greedy exploration. You can expect slightly higher rewards when rolling out the policies without any exploration at all.

| env  |  RLlib Basic DQN | RLlib Dueling DDQN | RLlib Distributional DQN  |  Hessel et al. DQN |  Hessel et al. Rainbow |
|---|---|---|---|---|---|
|BeamRider|2869|1910|4447|~2000|~13000|
|Breakout|287|312|410|~150|~300|
|QBert|3921|7968|15780|~4000|~20000|
|SpaceInvaders|650|1001|1025|~500|~2000|

Basic DQN plots:
![tensorboard](/atari-dqn/basic-dqn.png)

Dueling DDQN plots:
![tensorboard](/atari-dqn/dueling-ddqn.png)

Distributional DQN plots:
![tensorboard](/atari-dqn/dist-dqn.png)

#### Proximal Policy Optimization

`rllib train -f atari-ppo/atari-ppo.yaml`

`rllib train -f halfcheetah-ppo/halfcheetah-ppo.yaml`

##### *2018-09:*
RLlib PPO with 10 workers (5 envs per worker) after 10M and 25M time-steps 
(**40M/100M frames**). Note that RLlib does not use clip parameter annealing.

|env|RLlib PPO @10M|RLlib PPO @25M|Baselines PPO @10M|
|---|---|---|---|
|BeamRider|2807|4480|~1800|
|Breakout|104|201|~250|
|QBert|11085|14247|~14000|
|SpaceInvaders|671|944|~800|

![tensorboard](/atari-ppo/2018-09/atari-ppo.png)

RLlib PPO wall-time performance vs other implementations using a single Titan XP and the same number of CPUs. Results compared to learning curves from [Fan et al, 2018](https://surreal.stanford.edu/img/surreal-corl2018.pdf) extracted at 1 hour of training from Figure 7. Here we get optimal results with a vectorization of 32 environment instances per worker:

|env|RLlib PPO 16-workers|Fan et al PPO 16-workers|TF BatchPPO 16-workers|
|---|---|---|---|
|HalfCheetah|9664|~7700|~3200|

![tensorboard](/halfcheetah-ppo/halfcheetah-ppo.png)

##### *2020-01:*

Same as 2018-09, comparing only RLlib PPO-tf vs PPO-torch.

|env|RLlib PPO @20M (tf)|RLlib PPO @20M (torch)|plot|
|---|---|---|---|
|BeamRider|4142|3850|![tensorboard](/atari-ppo/BeamRiderNoFrameskip-v4/episode_reward_mean_tf_vs_torch_timesteps.png)|
|Breakout|132|166|![tensorboard](/atari-ppo/BreakoutNoFrameskip-v4/episode_reward_mean_tf_vs_torch_timesteps.png)|
|QBert|7987|14294|![tensorboard](/atari-ppo/QbertNoFrameskip-v4/episode_reward_mean_tf_vs_torch_timesteps.png)|
|SpaceInvaders|956|1016|![tensorboard](/atari-ppo/SpaceInvadersNoFrameskip-v4/episode_reward_mean_tf_vs_torch_timesteps.png)|

#### Soft Actor Critic

`rllib train -f halfcheetah-sac/halfcheetah-sac.yaml`

RLlib SAC after 3M time-steps. 

RLlib SAC versus SoftLearning implementation [Haarnoja et al, 2018](https://arxiv.org/pdf/1801.01290.pdf) benchmarked at 500k and 3M timesteps respectively.

|env|RLlib SAC @500K|Haarnoja et al SAC @500K|RLlib SAC @3M|Haarnoja et al SAC @3M|
|---|---|---|---|---|
|HalfCheetah|9000|~9000|13000|~15000|

![tensorboard](/halfcheetah-sac/halfcheetah-sac.PNG)

#### MAML

MAML uses additional metrics to measure performance; `episode_reward_mean` measures the agent's returns before adaptation, `episode_reward_mean_adapt_N` measures the agent's returns after N gradient steps of inner adaptation, and `adaptation_delta` measures the difference in performance before and after adaptation.

`rllib train -f maml/halfcheetah-rand-direc-maml.yaml`

![tensorboard](/maml/halfcheetah-rand-direc.png)

`rllib train -f maml/ant-rand-goal-maml.yaml`

![tensorboard](/maml/ant-rand-goal.png)

`rllib train -f maml/pendulum-mass-maml.yaml`

![tensorboard](/maml/pendulum-mass.png)

#### MB-MPO

`rllib train -f mbmpo/halfcheetah-mbmpo.yaml`

`rllib train -f mbmpo/hopper-mbmpo.yaml`

MBMPO uses additional metrics to measure performance. For each MBMPO iteration, MBMPO samples fake data from the transition dynamics workers and steps through MAML for `N` iterations. `MAMLIter$i$_DynaTrajInner_$j$_episode_reward_mean` corresponds to agent's performance across the dynamics models at the `i`th iteration of MAML and the `j`th step of inner adaptation.

RLlib MBMPO versus [Clavera et al, 2018](https://arxiv.org/pdf/1809.05214.pdf) benchmarked at 100k timesteps. Results reported below were ran on RLLib and the master branch of the [original codebase](https://github.com/jonasrothfuss/model_ensemble_meta_learning) respectively.

|env|RLlib MBPO @100K|Clavera et al MBMPO @100K|
|---|---|---|
|HalfCheetah|520|~550|
|Hopper|620|~650|

![tensorboard](/mbmpo/mbmpo-mujoco.png)

#### Dreamer

`rllib train -f dreamer/dreamer-deepmind-control.yaml`

RLlib Dreamer at 1M time-steps. 

RLlib Dreamer versus Google implementation [Danijar et al, 2020](https://arxiv.org/pdf/1912.01603.pdf) benchmarked at 100k and 1M timesteps respectively.

|env|RLlib Dreamer @100K|Danijar et al Dreamer @100K|RLlib Dreamer @1M|Danijar et al Dreamer @1M|
|---|---|---|---|---|
|Walker|320|~250|920|~930|
|Cheetah|300|~250|640|~800|

![tensorboard](/dreamer/deepmind-dreamer.png)

RLlib Dreamer also logs gifs of Dreamer's imagined trajectories (Top: Ground truth, Middle: Model prediction, Bottom: Delta).

![Alt Text](/dreamer/walker_dreamer.gif) ![Alt Text](/dreamer/halfcheetah_dreamer.gif)
