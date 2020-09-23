# Time: 2019-11-05  
## Desciption:  
This project is created for MADDPG, which is already popular in multi-agents. With the population of Pytorch, I think a version of pytorch for this project is useful for learners in multi-agents(Not for profit).   

## Installation
known dependencies: Python(3.6.8), OpenAI Gym(0.10.5), Pytorch(1.1.0), Numpy(1.17.3)    
Step 1: Install the MPE(Multi-Agent Particle Environments) as the readme of OpenAI (or the blog of mine).    
Step 2: Download the project and cd to this project. Make sure that you have a '\logs' and '\models' folder in your projects.    
Step 3: Run the 'main_openai.py' with 'python main_openai.py --scenario_name simple_tag'    

## Structure
./main_openai:		Main func    
./arguments.py: 	Init the par for game, training and saving.    
./model.py: 		Init the model for the agent.    
./replay_buffer.py:	Save the memory for all the agents.    
./enjoy_split.py: 	A templete for testing the model trained in the 'main_openai.py'.    

## Command line options
### Environment options
--scenario_name: defines which environment in the MPE is to be used (default: "simple")
--per_episode_max_len: maximum length of each episode for the environment (default: 45)
--max_episode: total number of training episodes (default: 150000)

### Training options
--lr_a: learning rate for actor(default: 1e-2)
--lr_c: learning rate for critic(default: 1e-2)
--gamma: discount factor (default: 0.95)
--batch_size: batch size (default: 1024)
--num_units_openai: number of units in the MLP (default: 64)

### Checkpointing
--save_dir: directory where intermediate training results and model will be saved (default: "/tmp/policy/")
--fre4save_model: model is saved every time this number of game steps has been completed (default: 1000)
--start_save_model: the time when we start to save the model(default: 400)

## Link for blog
OpenAI MPE: https://github.com/openai/multiagent-particle-envs    
The explorer of Game AI: https://zhuanlan.zhihu.com/c_186658689    

