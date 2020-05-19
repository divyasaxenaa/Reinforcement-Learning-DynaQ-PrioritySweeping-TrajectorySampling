import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy


class Maze:
    def __init__(self):
        self.GRID_WIDTH = 15
        self.GRID_HEIGHT = 25
        self.ACTION_UP = 0
        self.ACTION_DOWN = 2
        self.ACTION_LEFT = 1
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.START_STATE = [2, 0]
        self.GOAL_STATES = [[0, 8]]
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.q_size = (self.GRID_HEIGHT, self.GRID_WIDTH, len(self.actions))
        self.max_steps = 10000
        

    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.GRID_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.GRID_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 100.0
        elif [x, y] in self.obstacles:
            reward = -100.0
        else:
            reward = 0.0
        return [x, y], reward


class PriorityParams:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 0.1
        self.alpha = 0.1
        self.planning_steps = 5
        self.runs = 10
        self.theta = 0


class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder


class DynaQModel:
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward


class PriorityModel(DynaQModel):
    def __init__(self, rand=np.random):
        DynaQModel.__init__(self, rand)
        self.priority_queue = PriorityQueue()
        self.predecessors = dict()

    def insert(self, priority, state, action):
        self.priority_queue.add_item((tuple(state), action), -priority)

    def empty(self):
        return self.priority_queue.empty()

    def sample(self):
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return -priority, list(state), action, list(next_state), reward

    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        DynaQModel.feed(self, state, action, next_state, reward)
        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        self.predecessors[tuple(next_state)].add((tuple(state), action))

    def predecessor(self, state):
        if tuple(state) not in self.predecessors.keys():
            return []
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[tuple(state)]):
            predecessors.append([list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors



def __probability_given_state_action_orientation(state,next_state, orientation, action, rwrd):
    #------------------------------ ACTION FORWARD---------------------------------------
    if (action == 0):# action Forward
            new_orientation = 0
            if (rwrd == -100):
                new_orientation = orientation
    #------------------------------ ACTION TURN LEFT---------------------------------------
    if (action == 2): # action turn left
            new_orientation = 2
            if (rwrd == -100):
                new_orientation = orientation
    #------------------------------ ACTION BACKWARD---------------------------------------
    if (action == 1): # action backward
            new_orientation = 1
            if (rwrd == -100):
                new_orientation = orientation
    #------------------------------ ACTION TURN RIGHT---------------------------------------
    if (action == 3): # action turn right
            new_orientation = 3
            if (rwrd == -100):
                new_orientation = orientation
    if (next_state != state) and (action == 2 or action == 0):
        p = 0.8
    elif ( next_state == state) and (action == 2 or action == 0):
        p = 0.2
    elif (orientation != new_orientation ) and (action == 3 or action == 1):
        p = 0.9
    elif (new_orientation == orientation ) and (action == 3 or action == 1):
        p = 0.1
    return p,new_orientation


def __choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


def __print_prioritized_sweeping(q_value, model, maze, dyna_params):
    state = maze.START_STATE
    steps = 0
    orientation = 0
    while state not in maze.GOAL_STATES:
        steps += 1
        action = __choose_action(state, q_value, maze, dyna_params)
        next_state, reward = maze.step(state, action)
        dyna_params.gamma,new_orientation = __probability_given_state_action_orientation(state,next_state, orientation, action, reward)
        model.feed(state, action, next_state, reward)
        priority = np.abs(reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                          q_value[state[0], state[1], action])
        if priority > dyna_params.theta:
            model.insert(priority, state, action)
        planning_step = 0
        while planning_step < dyna_params.planning_steps and not model.empty():
            priority, state_, action_, next_state_, reward_ = model.sample()
            delta = reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - \
                    q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += dyna_params.alpha * delta
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + dyna_params.gamma * np.max(q_value[state_[0], state_[1], :]) -
                                  q_value[state_pre[0], state_pre[1], action_pre])
                if priority > dyna_params.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_step += 1
        state = next_state
        orientation = new_orientation

    return steps


def __Priority_Plot():
    dyna_maze = Maze()
    dyna_params = PriorityParams()
    runs = 10
    episodes = 100
    planning_steps = [0, 5, 50]
    steps = np.zeros((len(planning_steps), episodes))
    for run in tqdm(range(runs)):
        for i, planning_step in enumerate(planning_steps):
            dyna_params.planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)
            model = PriorityModel()
            for ep in range(episodes):
                steps[i, ep] += \
                    __print_prioritized_sweeping(q_value, model, dyna_maze, dyna_params)
    steps /= runs
    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], label='%d planning steps for Prioritized Sweeping' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.savefig('Priority_Plot.png')
    plt.close()


if __name__ == '__main__':
    __Priority_Plot()
