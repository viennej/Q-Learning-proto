from keras.models import Model
from keras.layers import Dense,Input
from keras.optimizers import Adam


class Agent(object):

    def __init__(self,learning_rate=0.001,num_choices=5):

        self.learning_rate = learning_rate
        inp = Input(shape=(3,))
        x = Dense(16,  activation='sigmoid')(inp)
        x = Dense(10, activation='sigmoid')(x)
        y = Dense(num_choices, activation='softmax')(x)

        # 0 - -10
        # 1 - -5
        # 2 - 0
        # 3 - +5
        # 4 - +10


        self.model = Model(inputs=inp,outputs=y)

        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

