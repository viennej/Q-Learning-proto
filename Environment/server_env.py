import numpy as np


class Server(object):
    # A class that models a server

    """
    Attributes of the server
    ---------------------------

       Constants
       -----------
       Optimum temperature range - core_temp_opt_range - Tuple (min,max) in degree C
       Absolute maximum temperature - core_temp_max
       Absolute minimum temperature - core_temp_min

       Month when simulation is started - start_month
       Max Number of users that can go up or down per minute - num_users_update
       Max Data transmission that can change per minute - rdt_update

       Lower threshold for number of users - num_users_lt
       Lower threshold for data transmission - rdt_lt

       Maximum number of users - max_users_ht
       Maximum rdt - max_rdt

       Initial number of users - initial_n_users

       Variables
       ---------
       Atmospheric temperature - atm_temp

       Temperature of core - core_temp

       Number of users at time in the server - n_users

       Rate of data transmission - rdt

       Energy spent by AI - energy

       Total energy spent - energy_tot

       Total energy spent without the agent - energy_wo_agent_tot

       Score per update - score
       Total score -  score_tot

       status = 1 -> If server is active
       status = 0 -> If server is dead (if temp goes out of extremes)

    """

    avg_temp_month = [1.0, 5.0, 7.0, 10.0, 11.0, 20.0, 23.0, 24.0, 22.0, 10.0, 5.0, 1.0]

    atm_temp = avg_temp_month[0]

    # Number of users
    num_users_update = 5
    num_users_lt = 10
    num_users_ht = 100
    num_users_ptp = num_users_ht - num_users_lt

    initial_n_users = num_users_lt
    n_users = initial_n_users

    # RDT
    rdt_update = 10
    rdt_lt = 20
    rdt_ht = 300
    rdt_ptp = rdt_ht - rdt_lt

    initial_rdt = rdt_lt
    rdt = initial_rdt

    effect_users = 0.1
    effect_data = 0.1

    core_temp = atm_temp + n_users * effect_users + rdt * effect_data
    maintained_core_temp = core_temp
    core_temp_max = 80
    core_temp_min = -20

    power_expended_ptp = core_temp_max - core_temp_min

    max_change = core_temp_max - core_temp_min

    energy = 0.0
    energy_tot = 0.0
    energy_wo_agent_tot = 0.0

    score = 0.0
    score_tot = 0.0

    game_over = 0

    # train or run
    train = 1

    def __init__(self, optimum_temp=(20.0, 22.0), start_month=0, initial_n_users=10, initial_rdt=60):
        # INITIALIZE THE SERVER

        assert (initial_n_users >= self.initial_n_users)
        self.initial_n_users = initial_n_users

        assert (initial_rdt >= self.initial_rdt)
        self.initial_rdt = initial_rdt

        self.core_temp_opt = optimum_temp
        self.start_month = start_month
        self.atm_temp = self.avg_temp_month[start_month]

        self.n_users = initial_n_users
        self.rdt = initial_rdt

        self.core_temp = self.atm_temp + self.n_users * self.effect_users + self.rdt * self.effect_data
        self.maintained_core_temp = self.core_temp

    def update_env(self, action, power_expended, month):

        # print "#########################"
        # DO THE ACTION

        flash_required = 0
        if (self.maintained_core_temp < self.core_temp_opt[0]):
            flash_required = self.core_temp_opt[0] - self.maintained_core_temp
            sign_flash_required = 1
        elif (self.maintained_core_temp > self.core_temp_opt[1]):
            flash_required = self.maintained_core_temp - self.core_temp_opt[1]
            sign_flash_required = -1

        if (action == 0):
            temp_change_from_action = -power_expended
        elif (action == 1):
            temp_change_from_action = power_expended

        # UPDATE THE ENERGY STATS
        self.energy = power_expended
        self.energy_tot += power_expended
        self.energy_wo_agent_tot += flash_required

        # GET THE SCORE TO BE RETURNED. SCORE - Energy saved - Performance
        self.score = 1e-3*(abs(abs(flash_required)-power_expended)) - 0.1e-3*(abs(flash_required))
        # self.score = self.weight_power * (flash_required - temp_change_from_action) - self.weight_perf * (flash_required)
        self.score_tot += self.score

        # UPDATE THE ENV
        new_n_users = self.n_users + np.random.randint(-self.num_users_update, self.num_users_update)

        if (new_n_users > self.num_users_ht):
            new_n_users = self.num_users_ht
        elif (new_n_users < self.num_users_lt):
            new_n_users = self.num_users_lt

        new_rdt = self.rdt
        if (new_rdt > self.rdt_ht):
            new_rdt = self.rdt_ht
        elif (new_rdt < self.rdt_lt):
            new_rdt = self.rdt_lt

        self.n_users = new_n_users
        self.rdt = new_rdt

        self.atm_temp = self.avg_temp_month[month%12]
        core_temp_past = self.core_temp
        self.core_temp = self.atm_temp + self.n_users * self.effect_users + self.rdt * self.effect_data
        core_temp_delta = self.core_temp - core_temp_past

        self.maintained_core_temp = self.maintained_core_temp + core_temp_delta + temp_change_from_action

        # print "MAINTAINED CORE TEMP:", self.maintained_core_temp
        # print "SCORE:", self.score

        if (self.maintained_core_temp > self.core_temp_max or self.maintained_core_temp < self.core_temp_min):
            # print "GAME OVER"
            if (self.train == 1):
                self.game_over = 1
            else:
                self.energy_tot += flash_required
                self.maintained_core_temp = self.maintained_core_temp + sign_flash_required * flash_required

        # Return the maintained_core_temp, the score, and the game status.
        scaled_core_temp = (self.maintained_core_temp - self.core_temp_min) / (
            self.core_temp_max - self.core_temp_min + 0.0)
        scaled_n_users = (self.n_users - self.num_users_lt) / (self.num_users_ht - self.num_users_lt)
        scaled_rdt = (self.rdt - self.rdt_lt) / (self.rdt_ht - self.rdt_lt)

        scaled_states = np.matrix([scaled_core_temp, scaled_n_users, scaled_rdt])

        return scaled_states, self.maintained_core_temp, self.score, self.game_over

    def reset(self,new_month):
        self.atm_temp = self.avg_temp_month[new_month]
        self.start_month = new_month

        self.n_users = self.initial_n_users
        self.rdt = self.initial_rdt

        self.core_temp = self.atm_temp + self.n_users * self.effect_users + self.rdt * self.effect_data
        self.maintained_core_temp = self.core_temp
        self.game_over = 0
        self.energy = 0
        self.energy_tot = 0
        self.energy_wo_agent_tot = 0
        self.score_tot = 0
        self.score = 0
        self.train = 1

    def observe(self):
        scaled_core_temp = (self.maintained_core_temp - self.core_temp_min) / (
            self.core_temp_max - self.core_temp_min + 0.0)
        scaled_n_users = (self.n_users - self.num_users_lt) / (self.num_users_ptp)
        scaled_rdt = (self.rdt - self.rdt_lt) / (self.rdt_ptp)

        scaled_states = np.matrix([scaled_core_temp, scaled_n_users, scaled_rdt])

        return scaled_states, self.maintained_core_temp, self.score, self.game_over
