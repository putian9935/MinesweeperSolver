import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Reveal: 
    def __init__(self, solver): 
        self.state = solver
    
        self.pix = solver.pix 
        self.horz_cent = solver.horz_cent 
        self.vert_cent = solver.vert_cent 
        self.vert_dim = solver.vert_dim 
        self.horz_dim = solver.horz_dim
        self.state = solver.state
    

    def _revealList(self, coord_lst, color='r'):
        _, ax = plt.subplots()
        ax.imshow(self.pix, interpolation=None)
        for i, j in coord_lst:
            rect = patches.Rectangle(( self.horz_cent[j] + self.horz_dim/4,self.vert_cent[i] + self.vert_dim/4,), self.vert_dim/2, self.horz_dim/2, facecolor=color, alpha=.5)
            ax.add_patch(rect)
        plt.show()


    def _revealMap(self):
        plt.figure()
        plt.imshow(self.pix, interpolation=None)
        for val in self.horz_cent:
            plt.axvline(val)
        for val in self.vert_cent:
            plt.axhline(val)
        plt.show()

    def _revealMines(self):
        lst = []
        for i in range(len(self.state)):
            for j in range(len(self.state[0])):
                if self.state[i, j] == -1:
                    lst.append((i, j))

        self._revealList(lst)

if __name__ == '__main__':
    import numpy as np 
    reveal = Reveal(np.random.randint(-2,3, size=(30,16))) 

    reveal._revealMines()