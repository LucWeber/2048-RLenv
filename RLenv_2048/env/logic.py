import random

from RLenv_2048.env import constants as c

def new_game(n):
    mat = []
    for i in range(n):
        mat.append([0] * n)

    mat = add_two(mat)
    mat = add_two(mat)
    return mat

def add_two(mat):
    a = random.randint(0, len(mat)-1)
    b = random.randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    mat[a][b] = 2
    return mat


def game_state(mat):
    # check for any zero entries
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
    # check for same cells that touch each other
    for i in range(len(mat)-1):
        # intentionally reduced to check the row on the right and below
        # more elegant to use exceptions but most likely this will be their solution
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new


def cover_up(mat):
    new = []
    for j in range(c.GRID_LEN):
        partial_new = []
        for i in range(c.GRID_LEN):
            partial_new.append(0)
        new.append(partial_new)
    action_executed = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    action_executed = True
                count += 1
    return new, action_executed


def merge(mat, action_executed):
    score = 0
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                score += mat[i][j]
                action_executed = True
    return mat, action_executed, score


def up(mat):
    # return matrix after shifting up
    mat = transpose(mat)
    mat, action_executed = cover_up(mat)
    mat, action_executed, score = merge(mat, action_executed)
    mat = cover_up(mat)[0]
    mat = transpose(mat)
    return mat, action_executed, score


def down(mat):
    mat = reverse(transpose(mat))
    mat, action_executed = cover_up(mat)
    mat, action_executed, score = merge(mat, action_executed)
    mat = cover_up(mat)[0]
    mat = transpose(reverse(mat))
    return mat, action_executed, score


def left(mat):
    # return matrix after shifting left
    mat, action_executed = cover_up(mat)
    mat, action_executed, score = merge(mat, action_executed)
    mat = cover_up(mat)[0]
    return mat, action_executed, score


def right(mat):
    # return matrix after shifting right
    mat = reverse(mat)
    mat, action_executed = cover_up(mat)
    mat, action_executed, score = merge(mat, action_executed)
    mat = cover_up(mat)[0]
    mat = reverse(mat)
    return mat, action_executed, score
