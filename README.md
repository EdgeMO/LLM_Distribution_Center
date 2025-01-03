# LLM_Distribution_Center
code implementation of center
1. establish the connection between edge node and center, register all the edge node and keep the connection alive for the whole process
2. get input data from local enviroment and generate input for each round
3. 20 task offloading rounds before model offloading round
4. after algorithem finished, match the target edge id with task offloading decisions and model caching decisions
5. call the transmission function and distribute the specific tasks to edge node
6. call the function for model transmission
7. after model deployment ready, ready for next round