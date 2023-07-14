import torch
import random
import numpy as np
from collections import deque
from agent import Agent
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class StudentAgent(Agent):

    def __init__(self):
        Agent.__init__(self)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_straight = ((dir_r and game.is_collision(point_r)) or
                           (dir_l and game.is_collision(point_l)) or
                           (dir_u and game.is_collision(point_u)) or
                           (dir_d and game.is_collision(point_d)))

        danger_right = ((dir_u and game.is_collision(point_r)) or
                        (dir_d and game.is_collision(point_l)) or
                        (dir_l and game.is_collision(point_u)) or
                        (dir_r and game.is_collision(point_d)))

        danger_left = ((dir_d and game.is_collision(point_r)) or
                       (dir_u and game.is_collision(point_l)) or
                       (dir_r and game.is_collision(point_u)) or
                       (dir_l and game.is_collision(point_d)))

        # TODO: Construct State
        state = [
            # Danger straight
            danger_straight,

            # Danger right
            danger_right,

            # Danger left
            danger_left,

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = StudentAgent()
    game = SnakeGameAI()
    while True:
        # TODO: 1. get old state
        state_old = agent.get_state(game)

        # TODO: 2. get move
        final_move = agent.get_action(state_old)

        # TODO: 3. perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # TODO: 4. train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        # TODO: 5. remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # TODO: 6. train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > agent.record:
                agent.record = score
                agent.model.save(agent.record)

            print('Game', agent.n_games, 'Score',
                  score, 'Record:', agent.record)

            # plot result
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
