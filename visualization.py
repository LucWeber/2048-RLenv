import sys
sys.path.insert(0, '/home/lucas/Desktop/2048-python')
import logic
import constants as c
from tkinter import Frame, Label, CENTER
import numpy as np
from time import sleep

class GridVisualization(Frame):
    def __init__(self, env, policy, sleep_time):
        Frame.__init__(self)
        self.policy = policy
        self.env = env
        self.obs = env.reset()
        self.sleep_time = sleep_time
        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

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

    def key_down(self, event):
        action, _ = self.policy.predict(np.asarray(self.obs).flatten())
        obs, rewards, done, info = self.env.step(action)
        self.update_grid_cells(self.env.matrix)
        #self.env.render(mode=mode, grid_visualization=self)
        if done:
            self.grid_cells[1][1].configure(text="Lost", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text=f"{self.env.total_score}", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            print(self.env.total_score)
            sleep(3)
            self.destroy()
        else:
            sleep(self.sleep_time)
            self.event_generate("<Key>")
