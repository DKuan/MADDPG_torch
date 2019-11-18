# Time: 2019-11-05  
## Desciption:  
This project is created for MADDPG, which is already popular in multi-agents.  

### Simple-adversary  
This scene is more difficult than simple.py, which we should set the model input for actor : 8(+2), critic: 8(+2)*1+5*3(obs+3*action_size)  
The reason for +2 is calculating the distance between the agent and the target   
