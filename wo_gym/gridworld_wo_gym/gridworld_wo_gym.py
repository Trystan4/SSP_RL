import random
import matplotlib.pyplot as plt
import numpy as np

secure_random = random.SystemRandom()
class GridWorld4x4:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3

    ACTIONS = [ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP]

    ACTION_NAMES = ["UP", "LEFT", "DOWN", "RIGHT"]

    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0)
    }

    num_actions = len(ACTIONS)

    def __init__(self, n, m, wrong_action_p=0.1, alea=False):
        self.n = n
        self.m = m
        self.wrong_action_p = wrong_action_p
        self.alea = alea
        self.generate_game()



    def _position_to_id(self, x, y):
        """Donne l'identifiant de la position entre 0 et 15"""
        return x + y * self.n

    def _id_to_position(self, id):
        """Réciproque de la fonction précédente"""
        return (id % self.n, id // self.n)

    def generate_game(self):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]
        hole = secure_random.choice(cases)
        cases.remove(hole)
        start = secure_random.choice(cases)
        cases.remove(start)
        end = secure_random.choice(cases)
        cases.remove(end)
        block = secure_random.choice(cases)
        cases.remove(block)

        self.position = start
        self.end = end
        self.hole = hole
        self.block = block
        self.counter = 0
        
        if not self.alea:
            self.start = start
        return self._get_state()
    
    def reset(self):
        if not self.alea:
            self.position = self.start
            self.counter = 0
            return self._get_state()
        else:
            return self.generate_game()

    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille

    def _get_state(self):
        if self.alea:
            return [self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.hole, self.block]]
        return self._position_to_id(*self.position)

    def move(self, action):
        """
        takes an action parameter
        :param action : the id of an action
        :return ((state_id, end, hole, block), reward, is_final, actions)
        """
        
        self.counter += 1

        if action not in self.ACTIONS:
            raise Exception("Invalid action")

        # random actions sometimes (2 times over 10 default)
        choice = secure_random.random()
        if choice < self.wrong_action_p:
            action = (action + 1) % 4
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 4

        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        if self.block == (new_x, new_y):
            return self._get_state(), -1, False, self.ACTIONS
        elif self.hole == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), -10, True, None
        elif self.end == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), 10, True, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), -1, False, self.ACTIONS
        elif self.counter > 190:
            self.position = new_x, new_y
            return self._get_state(), -10, True, self.ACTIONS
        else:
            self.position = new_x, new_y
            return self._get_state(), -1, False, self.ACTIONS

    def print(self):
        stri = ""
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if (i, j) == self.position:
                    stri += "R"
                elif (i, j) == self.block:
                    stri += "B"
                elif (i, j) == self.hole:
                    stri += "O"
                elif (i, j) == self.end:
                    stri += "E"
                else:
                    stri += "."
            stri += "\n"
        print(stri)

    def choose_action(self, q_table,etat,espace,epsilon):
        print(espace)
        a_opt=np.argmax(secure_random.shuffle(q_table[etat]))
        if (secure_random.random() > epsilon):
            return int(a_opt)
        else:
            action=secure_random.sample(espace, 1)
            while (action==a_opt):
                action=secure_random.sample(espace, 1)
            return action[0]
    
    def q_learning(self,states_n,actions_n, beta, epsilon, num_episodes):
        ' Tableau q_table(s,a)'
        q_table = np.zeros([states_n, actions_n])
        ' Tableau N(s,a)'
        N=np.zeros([states_n, actions_n])

        cumul_reward_list = []
        actions_list = [0,1,2,3]
        states_list = []
        grid = self(4, 4, 0) # 0.1 chance to go left or right instead of asked direction
        for i in range(num_episodes):
            actions = []
            s = grid.reset()
            states = [s]
            cumul_reward = 0
            d = False
            while True:
                obs_c=states
                #print("Etat courant",obs_c)
                a= grid.choose_action( q_table,obs_c,actions_list,epsilon)
                #print("action",a)
                #print("Etat suivant",observation)

                s1, reward, d, _ = grid.move(a)

                ' mise a jour alpha'
                N[obs_c,a]=int(N[obs_c,a]+1)
                alpha=1/N[obs_c,a]
                ' recuperation action optimale'
                a_opt=np.argmax(q_table[obs_c])
                #print("action Optimale",a_opt)
                ' mise a jour q_table table'
                q_table[obs_c,a]=(1-alpha)*q_table[obs_c,a]+ alpha*(reward +beta*q_table[s1,a_opt])
                
                # # probability to take a random action
                # Q2 = q_table[s,:] + np.random.randn(1, actions_n)*(1. / (i +1))
                # a = np.argmax(Q2)
                # q_table[s, a] = q_table[s, a] + lr*(reward + y * np.max(q_table[s1,:]) - q_table[s, a]) # Fonction de mise à jour de la q_table-table

                cumul_reward += reward
                s = s1
                actions.append(a)
                states.append(s)
                if d == True:
                    break
            states_list.append(states)
            actions_list.append(actions)
            cumul_reward_list.append(cumul_reward)
            print(q_table)

            print("Construction de la politique")
            pi=np.zeros(states_n)

            for i in range(states_n):
                pi[i]=int(np.argmax(q_table[i]))

            print("politique=",pi)    
        grid.reset()
        grid.print()
        return q_table, cumul_reward_list