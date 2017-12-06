import numpy as np
import os
import random as rn

#Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

from Environment import agent
from Environment import server_env
from Qlearn import qlearn
import matplotlib.pyplot as plt
from keras.models import load_model

plt.ioff()
import matplotlib

matplotlib.use('Agg')

# parameters
epsilon = .3  # Exploration coefficient
num_choices = 5
epoch = 4000  # Number of epochs to be trained
max_memory = 3000  # Max memory of q-learning from which batches are selected
batch_size = 128  # Batch size of each training iteration

# Get the AI model
agent = agent.Agent(learning_rate=0.00001, num_choices=num_choices)
model = agent.model

# If you want to continue training from a previous model, just uncomment the line below and make Train=False
# model=load_model("model.h5")
train = True

# Define environment/game
a = 18.0
b = 24.0
env = server_env.Server(optimum_temp=(a, b), start_month=0, initial_n_users=20, initial_rdt=30)
optimum_temp_avg = (a + b) / 2.0

# Initialize experience replay object
exp_replay = qlearn.ExperienceReplay(max_memory=max_memory)

# Train
score = 0
score_list = []
if (train):
    for e in range(epoch):

        score = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        print new_month
        env.reset(new_month=new_month)
        game_over = False
        # get initial input
        input_t, x, y, z = env.observe()
        timestep = 0

        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):

            input_tm1 = input_t  # set input state

            # Get next action to be performed (Exploration or model choice for next action)
            if np.random.rand() <= epsilon:
                choice = np.random.randint(0, num_choices)
                if (choice - 2 < 0):
                    action = 0
                else:
                    action = 1

                power_expended = abs(choice - 2) * 5

            else:
                q = model.predict(input_tm1)
                choice = np.argmax(q[0])

                if (choice - 2 < 0):
                    action = 0
                else:
                    action = 1

                power_expended = abs(choice - 2) * 5

            # apply action, get rewards and new state
            input_t, inp, reward, game_over = env.update_env(action, power_expended, int(timestep / (30 * 24 * 60)))

            if reward > 0:
                score += reward

            # store experience
            exp_replay.remember([input_tm1, choice, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # Loss from the epoch updated
            loss += model.train_on_batch(inputs, targets)

            timestep += 1

        score_list.append(score)
        print("Epoch {:03d}/{:04d} | Loss {:.4f} | Win count {}".format(e, epoch, loss, score))
        print env.score_tot
        print env.energy_tot
        print env.energy_wo_agent_tot

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save("model.h5")
    # Make plot of the scores temperature
    temp_ai_score, = plt.plot(score_list, label='AI scores per epoch')
    plt.ylabel('AI scores per epoch')
    plt.xlabel('Epochs')
    plt.legend(handles=[temp_ai_score])
    plt.savefig('score_plot.png')
    plt.close()

print "START EXECUTION PHASE"
temp_list = []
temp_list_optimum_avg = []
env.train = 0
num_minutes_in_year = 365 * 24 * 60
num_minutes_in_month = 30 * 24 * 60

input_t, x, y, z = env.observe()


for i in range(0, num_minutes_in_year):
    month = np.int(round(num_minutes_in_year / (num_minutes_in_month))) % 12
    # print month
    q = model.predict(input_t)
    choice = np.argmax(q[0])

    if (choice - 2 < 0):
        action = 0
    else:
        action = 1

    power_expended = abs(choice - 2) * 5
    input_t, inp, reward, game_over = env.update_env(action, power_expended, month)
    temp_list.append(env.maintained_core_temp)
    temp_list_optimum_avg.append(optimum_temp_avg)

print env.score_tot
print env.energy_tot
print env.energy_wo_agent_tot

print "ENERGY SAVED %", ((env.energy_wo_agent_tot - env.energy_tot) / env.energy_wo_agent_tot) * 100

# Make plot of the server temperature
temp_ai, = plt.plot(temp_list, label='Temperature from AI')
temp_opt, = plt.plot(temp_list_optimum_avg, label='Optimum temperature')
plt.ylabel('Maintained core temperature')
plt.xlabel('Timesteps: minutes')
plt.legend(handles=[temp_ai, temp_opt])
plt.savefig('temp_plot.png')
plt.close()
