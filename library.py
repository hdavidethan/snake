import pygame


def is_left_key_pressed(event):
    return event.key == pygame.K_LEFT


def is_right_key_pressed(event):
    return event.key == pygame.K_RIGHT


def is_down_key_pressed(event):
    return event.key == pygame.K_DOWN


def is_up_key_pressed(event):
    return event.key == pygame.K_UP


def quit_game():
    pygame.quit()
    quit()
