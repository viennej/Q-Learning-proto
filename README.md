#Server AI
Initial implementation to test Q-Learning in python 2.7


Server: 
The Server environment we are using here is defined by some pre-definable parameters. This
includes the optimum temperature range of the server, the absolute maximum and minimum temperature beyond which the server fails to operate, maximum and minimum number of users the
server can host, the maximum and minimum rate of data transmission possible from the server,
etc.
The core temperature of the server is a function of the atmospheric temperature, the number of
users and the rate of data transmission. The number of users and the rate of data transmission is
randomly fluctuated to simulate an actual server. This leads to randomness in the core temperature and the AI has to understand how much cooling or heating power it has to spend so as to not
deteriorate the server performance and at the same time, expend the least energy by only doing
sufficient heat management

AI algo:
Here, we are using Q-learning to train the AI, the score to each action of the AI is given
by a linear combination of the distance of the core temperature is from the optimum temperature
and the power expended by the AI in a way that neither the performance of the server nor the
power spent is being sacrificed by the model.

Training:
The model is trained for 4000 runs (epochs) each one starting at a random month of the year (to
simulate the average atmospheric temperature). The exploration coefficient is set to 0.3 and the
max memory for q-learning batches is set to 3000. The batch size for each training is set to 128.
The training process also generates a graph of the number of scores per epoch that shows clearly
how the scores obtained by the AI increases on average on each passing epoch. This behavior is
very important to signify that the AI is learning something constructive.

Testing:
Once the model has been trained, it is saved into a local file. The model is now simulated for an
entire year with each minute of the year being treated as a time-step. Once the simulation is over,
we see that 47% energy is saved. (The energy saved is calculated as the difference between the
power expended to keep the temperature at the optimum and the energy spent by the AI)