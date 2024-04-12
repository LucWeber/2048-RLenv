import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from tkinter import Frame, Label, CENTER
import numpy as np
from time import sleep
from PIL import ImageGrab

from RLenv_2048.env import constants as c

class GridVisualization(Frame):
    def __init__(self, env, policy, sleep_time, title='2048'):
        Frame.__init__(self)
        self.policy = policy
        self.env = env
        self.obs = env.reset()
        self.sleep_time = sleep_time
        self.grid()
        self.master.title(title)
        self.master.bind("<Key>", self.key_down)
        self.iteration = 0
        self.record_screen = True

        self.history_matrices = []

        self.grid_cells = []
        self.init_grid()
        self.matrix = env.matrix
        self.update_grid_cells(self.matrix)

        self.event_generate("<<Foo>>", when="tail")
        self.mainloop()


    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def update_grid_cells(self, matrix):
        # Render the environment to the screen
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=c.BACKGROUND_COLOR_DICT[new_number],
                                                    fg=c.CELL_COLOR_DICT[new_number])
        self.update_idletasks()
        if self.record_screen:
            self.capture_screen()

    def capture_screen(self):
        x0 = self.winfo_rootx()
        y0 = self.winfo_rooty()
        x1 = x0 + self.winfo_width()
        y1 = y0 + self.winfo_height()
        ImageGrab.grab().crop((x0, y0, x1, y1)).save(f"./RLenv_2048/screenshots/game_state_{self.iteration}.png")
        self.iteration += 1

    def key_down(self, event):
        action = self.policy.predict(np.asarray(self.obs).flatten())
        obs, rewards, done, info = self.env.step(action)
        self.update_grid_cells(self.env.matrix)
        if done:
            self.display_end_game()
            sleep(3)
            self.destroy()
        else:
            sleep(self.sleep_time)
            self.event_generate("<Key>")

    def display_end_game(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                self.grid_cells[i][j].configure(text="")
        self.grid_cells[1][1].configure(text="Game Over", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text=f"Score: {self.env.total_score}", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
