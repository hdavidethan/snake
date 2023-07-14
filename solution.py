import pygame
import random
from enum import Enum
from collections import namedtuple
import library

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
# font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

    def update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def nextTick(self):
        self.clock.tick(SPEED)

    ########################################
    # YOU MAY USE THE CODE BELOW THIS LINE #
    ########################################

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                library.quit_game()
            if event.type == pygame.KEYDOWN:
                # TODO: Check for user inputs and change directions
                if library.is_down_key_pressed(event):
                    self.direction = Direction.DOWN if self.direction != Direction.UP else self.direction
                elif library.is_up_key_pressed(event):
                    self.direction = Direction.UP if self.direction != Direction.DOWN else self.direction
                elif library.is_left_key_pressed(event):
                    self.direction = Direction.LEFT if self.direction != Direction.RIGHT else self.direction
                elif library.is_right_key_pressed(event):
                    self.direction = Direction.RIGHT if self.direction != Direction.LEFT else self.direction

        # 2. TODO: Get the next head position
        game_over = False
        nextHeadPosition = self._get_next_head_position(self.direction)

        # 3. TODO: Check if next head position is a collision, food, or empty
        if self._is_collision(nextHeadPosition):
            game_over = True
            return game_over, self.score
        elif nextHeadPosition == self.food:
            self._eat(nextHeadPosition)
            self.score += 1
            self._place_food()
        else:
            self._move(nextHeadPosition)

        # 4. return game over and score
        return game_over, self.score

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _is_collision(self, position):
        # Check if a position is a collision
        # 1. TODO: hits boundary
        if self._hit_right_wall(position) or self._hit_left_wall(position) or self._hit_bottom_wall(position) or self._hit_top_wall(position):
            return True
        # 2. TODO: hits itself
        if self._self_collision(position):
            return True

        # 3. no collision
        return False

    def _get_next_head_position(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        return Point(x, y)

    def _update_head_position(self, position):
        self.head = position

    def _move(self, position):
        self._update_head_position(position)
        self.snake.insert(0, self.head)
        self.snake.pop()

    def _eat(self, position):
        self._update_head_position(position)
        self.snake.insert(0, self.head)

    def _hit_right_wall(self, position):
        return position.x > self.w - BLOCK_SIZE

    def _hit_left_wall(self, position):
        return position.x < 0

    def _hit_bottom_wall(self, position):
        return position.y > self.h - BLOCK_SIZE

    def _hit_top_wall(self, position):
        return position.y < 0

    def _self_collision(self, position):
        return position in self.snake[1:]


if __name__ == '__main__':
    game = SnakeGame()

    # game loop
    while True:
        game_over, score = game.play_step()

        game.update_ui()
        game.nextTick()

        if game_over == True:
            break

    print('Final Score', score)

    pygame.quit()
