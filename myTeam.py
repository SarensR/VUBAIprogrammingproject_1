# FIRSTTRY.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

# Caetano Team
#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DefensiveAgent', second='OffensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class DefensiveAgent(CaptureAgent):
    """
  class holding choose_action and methods to be used bij def and off
  """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.opponent_start = None

        # initialized values
        self.minimax_depth = 2
        self.status = 'patrolling_up'
        self.patrolling_distance = 2
        self.patrol_route = []

        self.border = 0
        self.opponents = []
        self.observable_opponents = []
        self.distances_opponents = []

    def register_initial_state(self, game_state):
        """
        registers initial state
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        # MAZE INFO
        width = game_state.data.layout.width - 2
        height = game_state.data.layout.height - 2
        self.border = int(width / 2)

        if self.red:
            self.opponent_start = (width, height)
        else:
            self.opponent_start = (1, 1)

        # DATA FILL IN
        self.opponents = self.get_opponents(game_state)

        # CALCULATIONS

        def find_patrol_points():
            if self.red:
                x_range = range(self.border, 1, -1)
            else:
                x_range = range(self.border, width)

            small_diff_bottom = 999
            small_diff_top = 999
            bottom_of_patrol = (self.border, 3)
            top_of_patrol = (self.border, height - 2)
            for x in x_range:
                if not game_state.has_wall(x, 3) and abs(self.patrolling_distance - abs(x - self.border)) < small_diff_bottom:
                    small_diff_bottom = self.patrolling_distance - abs(x - self.border)
                    bottom_of_patrol = (x, 3)
                if not game_state.has_wall(x, height - 2) and abs(self.patrolling_distance - abs(x - self.border)) < small_diff_top:
                    small_diff_top = self.patrolling_distance - abs(x - self.border)
                    top_of_patrol = (x, height - 2)
            return bottom_of_patrol, top_of_patrol

        self.patrol_route = find_patrol_points()

    def choose_action(self, game_state):
        """
        chase or patrol
        """
        # data fill in
        agent_state = game_state.get_agent_state(self.index)
        curr_pos = agent_state.get_position()
        self.update_observable_opponents_and_distances(game_state)
        if self.status == 'patrolling_down' and agent_state.get_position() == self.patrol_route[0]:
            self.status = 'patrolling_up'
        elif self.status == 'patrolling_up' and agent_state.get_position() == self.patrol_route[1]:
            self.status = 'patrolling_down'

        # chase or patrol
        if ((self.observable_opponents[0] and util.manhattanDistance(curr_pos, self.observable_opponents[0]) <= 5)
                or (self.observable_opponents[1] and util.manhattanDistance(curr_pos, self.observable_opponents[1]) <= 5)):
            return self.alpha_beta_minimax(game_state, self.minimax_depth, self.evaluation_function_defence)
        else:
            actions = game_state.get_legal_actions(self.index)
            if self.status == 'patrolling_down':
                patrol_point = self.patrol_route[0]
            else:
                patrol_point = self.patrol_route[1]
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(patrol_point, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def is_eaten(self, agent_index, game_state):
        if agent_index in self.opponents:
            return self.opponent_start == game_state.get_agent_position(agent_index)
        else:
            return self.start == game_state.get_agent_position(agent_index)

    def on_own_side(self, pos):
        """
        returns boolean indicating whether the position is on the agents own side.
        """
        if self.red:
            return pos[0] <= self.border
        else:
            return pos[0] > self.border

    def update_observable_opponents_and_distances(self, game_state):
        distances = game_state.get_agent_distances()
        self.observable_opponents = [False for i in range(game_state.get_num_agents())]
        for opponent in self.opponents:
            pos = game_state.get_agent_position(opponent)
            if pos:
                self.observable_opponents[opponent] = pos
                distances[opponent] = self.get_maze_distance(game_state.get_agent_position(self.index), pos)
            elif not pos and distances[opponent] <= 5:
                distances[opponent] = 6
        self.observable_opponents = [self.observable_opponents[index] for index in self.opponents]
        self.distances_opponents = [distances[index] for index in self.opponents]
    def evaluation_function_defence(self, game_state):
        """
        defensive evaluation function, expanded on DefensiveReflexAgent
        """
        features = util.Counter()
        agent_state = game_state.get_agent_state(self.index)
        pos = agent_state.get_position()
        self.update_observable_opponents_and_distances(game_state)

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if agent_state.is_pacman:
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        invaders = [observable_opponent for observable_opponent in self.observable_opponents if observable_opponent
                    and self.on_own_side(observable_opponent)]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            features['invader_distance'] = min(self.distances_opponents)

        # what to do when scared_timer is active?
        features['scared_timer'] = agent_state.scared_timer

        # factor in the score
        features['score'] = game_state.get_score()

        # defended food left
        own_food_list = self.get_food_you_are_defending(game_state).as_list()
        features['own_food_left'] = len(own_food_list)

        features['is_eaten'] = self.is_eaten(self.index, game_state)
        # distance between opponent and border
        # if invaders:
        #     features['dis_opponent_to_its_side'] = min([self.get_maze_distance(i, self.opponent_start) for i in invaders])

        weights = {'on_defense': 100, 'num_invaders': -10000, 'invader_distance': -30, 'scared_timer': -2,
                   'score': 2, 'own_food_left': 2, 'dis_opponent_to_its_side': -1}


        return features * weights

    def alpha_beta_minimax(self, game_state, depth, evaluation):
        caller = self.index
        opponent1 = False
        opponent2 = False
        old_minimax_depth = self.minimax_depth
        max_number_recursions = pow(4, 2 * self.minimax_depth) + 2
        num_rec = util.Counter()
        num_rec["recs"] = 0

        opponents = []
        for ind, pos in enumerate(self.observable_opponents):
            if pos and ind == 0:
                opponents.append(self.opponents[0])
            elif pos and ind == 1:
                opponents.append(self.opponents[1])

        if len(opponents) == 0:
            return util.raiseNotDefined()
        elif len(opponents) == 1:
            opponent1 = opponents[0]
        else:
            self.minimax_depth = 1
            if caller == 0 or caller == 3:
                opponent1, opponent2 = self.opponents
            else:
                opponent2, opponent1 = self.opponents

        def mini_max(game_state, agent, depth_minimax, a=-float('inf'), b=float('inf')):
            num_rec["recs"] += 1
            if num_rec["recs"] >= max_number_recursions:
                return ['action?', evaluation(game_state)]
            if (depth_minimax == 0
                    or self.is_eaten(agent, game_state)
                    or game_state.is_over()
                    or self.is_eaten(opponent1, game_state)):
                return ['action?', evaluation(game_state)]

            else:
                # make list [[action, value],[action,value], ...]
                scored_actions = []
                actions = game_state.get_legal_actions(agent)
                v = 0
                if agent == caller:
                    v = -v
                for action in actions:
                    new_state = game_state.generate_successor(agent, action)
                    if agent == caller:
                        minimax_max_value = mini_max(new_state, opponent1, depth_minimax, a, b)[1]
                        v = max(v, minimax_max_value)  # pseudocode van alpha beta
                        if v > b:
                            return [action, v]
                        else:
                            a = max(a, v)
                        scored_actions.append([action, minimax_max_value])
                    else:
                        if opponent2 and agent == opponent2:
                            minimax_min_value = mini_max(new_state, caller, depth_minimax - 1, a, b)[1]
                        elif opponent2 and agent == opponent1:
                            minimax_min_value = mini_max(new_state, opponent2, depth_minimax, a, b)[1]
                        elif not opponent2 and agent == opponent1:
                            minimax_min_value = mini_max(new_state, caller, depth_minimax - 1, a, b)[1]
                        else:
                            return util.raiseNotDefined()

                        v = min(v, minimax_min_value)
                        if v < a:
                            return [action, v]
                        else:
                            b = min(b, v)
                        scored_actions.append([action, minimax_min_value])

                # choose best from list
                best_scored_action = scored_actions[0]
                for scored_action in scored_actions:
                    if agent == caller:
                        if scored_action[1] > best_scored_action[1]:
                            best_scored_action = scored_action
                    if agent != caller:
                        if scored_action[1] < best_scored_action[1]:
                            best_scored_action = scored_action
                return best_scored_action

        return_value = mini_max(game_state, caller, depth, -float('inf'), float('inf'))[0]
        self.minimax_depth = old_minimax_depth
        return return_value
    def agenda_search(self, priority_selector, heuristic_function, game_state, start_pos, is_goal_state) -> list:

        agenda = util.PriorityQueueWithFunction(priority_selector)
        start_state = (start_pos, [], 0)  # turns the starting position into a triple.
        agenda.push(start_state)  # push the start_state on the agenda.
        get_successors = game_state.get_legal_actions
        already_visited = set()

        def position(successor_triple) -> tuple:
            return successor_triple[0]  # abstraction of the coordinate-selector of a successor

        def direction(successor_triple) -> str:
            return successor_triple[1]  # abstraction of the action-selector

        def step_cost(successor_triple) -> float:
            return successor_triple[2]  # abstraction of the step_cost-selector

        def heuristic_value(state) -> float:
            return heuristic_function(position(state))  # returns a float as the result of the heuristic function.

        while not agenda.isEmpty():
            current_state = agenda.pop()
            current_position = position(current_state)

            if current_position in already_visited:
                continue

            current_directions = direction(current_state)
            already_visited.add(current_position)

            if is_goal_state(current_position):
                return current_directions  # return the path from start_state to current_state as a list of actions.
            else:
                for successor in get_successors(current_position):
                    if position(successor) not in already_visited:
                        successor_directions = current_directions + [direction(successor)]
                        successor_step_cost = step_cost(current_state) - heuristic_value(current_state) + step_cost(
                            successor)
                        successor_utility = successor_step_cost + heuristic_value(successor)
                        agenda.push((position(successor), successor_directions, successor_utility))

    def a_star_search(self, heuristic_function, game_state, start_pos, is_goal_state):
        """Search the node that has the lowest combined cost and heuristic first."""
        return self.agenda_search(lambda state: state[2], heuristic_function, game_state, start_pos, is_goal_state)


class OffensiveAgent(DefensiveAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.food_eaten_count = 0  # Initialize the counter for food pellets eaten
        # Data fields
        self.current_dir = "String" # huidige directie
        self.current_pos = None # huidige positie
        self.is_this_a_pacman = False # of je een pacman bent of niet
        self.num_food_returned = 0 # het aantal food dat ik aan het retourneeren ben
        self.num_food_carrying = 0 #  het aantal dat ik aan het carryen ben
        self.scared_timer = 0 # timer dat zegt hoeveel tijd ik scared ben
        self.border = 0
        self.opponents = [] # lijst met opponents
        self.agents_observable = [] # oppponents dat ik kan zien
        self.ally = 0
        self.max_food_to_get = 5

    def register_initial_state(self, game_state):

        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # MAZE INFO
        width = game_state.data.layout.width - 2  # length = game_state.data.layout.height - 2
        self.border = width / 2

        # AGENTS
        self.opponents = self.get_opponents(game_state) # geeft mij de opponents in de game_State
        self.agents_observable = [False for i in range(game_state.get_num_agents())] # geeft mij de observable agents
        self.agents_observable[self.index] = game_state.get_agent_position(self.index) # geeft mij hun index , dus waar ze zijn
        self.ally = [agent for agent in self.get_team(game_state) if agent != self.index][0] # geeft mij nu mijn allies
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
    def updated_distances_list(self, game_state): # ROBBE
        distances = game_state.get_agent_distances()
        for agent in range(game_state.get_num_agents()):
            pos = game_state.get_agent_position(agent)
            if pos: # Als de lijst niet leeg is dus
                self.agents_observable[agent] = pos
                distances[agent] = self.get_maze_distance(game_state.get_agent_position(self.index), pos) # Neemt de afstand
            elif not pos and distances[agent] <= 5:
                distances[agent] = 6
        return distances

    def getOpponents(self, game_state):
        num_agents = len(game_state.data.agent_states)
        return [agent_index for agent_index in range(num_agents) if
                agent_index != self.index and not game_state.data.agent_states[agent_index].is_pacman]

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        agent_state = successor.get_agent_state(self.index)  # De state van een agent
        succ_scared_timer = agent_state.scared_timer   # Timer die stijgt naarmate de agent scared is
        succ_num_food_carrying = agent_state.num_carrying # het aantal eten dat een state van een agent heeft
        succ_num_food_returned = agent_state.num_returned # het aantal eten dat een agent terugbrengt
        succ_is_this_a_pacman = agent_state.is_pacman # of een agent een pacman is of niet
        succ_pos = agent_state.get_position()
        succ_dir = agent_state.get_direction()
        distances = self.updated_distances_list(successor)

        features['successor_score'] = self.get_score(successor)
        features['own_scared_timer'] = succ_scared_timer  # no feature but affects others?
        features['num_food_carrying'] = succ_num_food_carrying
        features['num_food_returned'] = succ_num_food_returned
        # successor_score: The score of the successor state after taking the given action.
        # own_scared_timer: The scared timer of the agent after taking the action.
        # num_food_carrying: The number of food items the agent is currently carrying.
        # num_food_returned: The number of food items the agent has returned to its side.

        food_list = self.get_food(successor).as_list()
        features['food_left'] = len(food_list)
        capsule_list = self.get_capsules(successor)
        features['capsule_left'] = len(capsule_list)
        if succ_pos in self.get_capsules(game_state): features['capsule_eaten'] = 1
        if len(capsule_list) > 0:
            features['dis_nearest_capsule'] = min([self.get_maze_distance(succ_pos, caps) for caps in capsule_list])

        if not self.is_this_a_pacman:
            features['dis_nearest_defender_not_scared'] = 1000

        if action == 'Stop':
            features['Stop'] = 1
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            features['dis_nearest_food'] = min([self.get_maze_distance(succ_pos, food) for food in food_list])
            # Distance to the nearest enemy
            # It would also be interesting to know where the closest enemy is
            # So we can compute the distance to the nearest enemy
            # def getOpponents(self, gameState):
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None ]
            not_scared_defenders = [enemy for enemy in defenders if enemy.scared_timer == 0]
            scared_defenders = [enemy for enemy in defenders if enemy.scared_timer > 0]

            if len(not_scared_defenders) > 0:
                minDist = min([self.get_maze_distance(succ_pos, defs.get_position()) for defs in not_scared_defenders])
                features['dis_nearest_defender_not_scared'] = minDist
                features['successor_score'] = -100
                features['num_food_carrying'] = -100
                features['dis_nearest_food'] =-100

            if len (scared_defenders) > 0:
                minDist = min([self.get_maze_distance(succ_pos, defs.get_position()) for defs in scared_defenders])
                features['dis_to_scared_defender'] = minDist

            features['dis_nearest_opponent'] = min([distances[i] for i in self.opponents])

            if len(defenders) > 0:
                minDistance = min([self.get_maze_distance(succ_pos, defs.get_position()) for defs in defenders])
                features['distanceToEnemy'] = minDistance
        return features

    def on_own_side(self, pos):
        if self.red:
            return pos[0] < self.border
        else:
            return pos[0] >= self.border

    def eat_normal(self, game_state,actions):
        actions = game_state.get_legal_actions(self.index)
        food_list = self.get_food(game_state).as_list()
        min_distance = 999
        best_action = None
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos = successor.get_agent_position(self.index)
            if pos in food_list:
                distance_to_food = self.get_maze_distance(self.current_pos, pos)
                if distance_to_food < min_distance:
                    min_distance = distance_to_food
                    best_action = action
        return best_action

    def min_distance_to_own_side(self, game_state, pos):
        if self.red:
            x = int(self.midWidth - 1)
        else:
            x = int(self.midWidth + 1)
        positions = [(x, y) for y in range(self.height) if not game_state.data.layout.is_wall((x, y))]
        if not positions:
            return 999
        distance_to_home = min(self.get_maze_distance(position, pos) for position in positions)
        return distance_to_home

    def choose_action(self, game_state):
        agent_state = game_state.get_agent_state(self.index)
        self.scared_timer = agent_state.scared_timer
        self.num_food_carrying = agent_state.num_carrying
        self.num_food_returned = agent_state.num_returned
        self.is_this_a_pacman = agent_state.is_pacman
        self.current_pos = agent_state.get_position()
        self.current_dir = agent_state.get_direction()
        succ_pos = agent_state.get_position()
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        if len(values) > 1: values = [a for a in values if a != 'Stop']  # ROBBE
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        food_left = len(self.get_food(game_state).as_list())

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        scared_defenders = [enemy for enemy in defenders if enemy.scared_timer > 0]

        if (food_left <= 2 or self.num_food_carrying >= 3) and scared_defenders == []:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                distance_to_own = self.min_distance_to_own_side(game_state, pos2)
                if distance_to_own < best_dist:
                    best_action = action
                    best_dist = distance_to_own
                if action == 'Stop':
                    best_action = best_action
            return best_action

        if scared_defenders and self.num_food_carrying < 7:
            best_action = self.eat_normal(game_state, actions)
            if best_action:
                return best_action






        return random.choice(best_actions)
    def get_weights(self, game_state, action):
        return {'successor_score': 10, 'own_scared_timer': -1, 'num_food_carrying': 100, 'num_food_returned': 100,
                'food_left': -1, 'capsule_left': 0, 'capsule_eaten': 10, 'dis_nearest_food': -1,
                'dis_nearest_defender_not_scared': 50, 'pacman_eaten': 10000, 'Stop': -1000, 'dis_to_scared_defender': -1000}

        # hou de dis_to_Scared_Def negatief om naar hem te gaan


