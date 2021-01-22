# from main import Solver 
import numpy as np 

class BruteForceSolver: 
    def __init__(self, solver):
        idx = 0 
        mapping = dict() 
        inv_mapping = dict()

        # first run, get indices right
        for i, j in solver.frontiers:
            for di, dj in [(0, 1), (0, -1), (1, 1), (1, -1),
                (-1, 1), (-1, -1), (-1, 0), (1, 0)]:
                cur_i = i + di 
                cur_j = j + dj
                if solver._checkUnopened(cur_i, cur_j):
                    if (cur_i, cur_j) not in mapping:
                        mapping[(cur_i, cur_j)] = idx 
                        inv_mapping[idx] = (cur_i, cur_j)
                        idx += 1
        
        coeffs = []
        b = []
        tot_unknown = len(mapping)
        # second run, get coeffs right 
        for i, j in solver.frontiers:
            new = np.zeros(tot_unknown, dtype=np.int32)
            mines = 0
            for di, dj in [(0, 1), (0, -1), (1, 1), (1, -1),
                (-1, 1), (-1, -1), (-1, 0), (1, 0)]:
                cur_i = i + di 
                cur_j = j + dj
                if solver._checkUnopened(cur_i, cur_j):
                    new[mapping[(cur_i, cur_j)]] = 1 
                if solver._checkMine(cur_i, cur_j):
                    mines += 1
            if np.any(new):
                coeffs.append(new) 
                b.append(solver.state[i, j] - mines)
        
        coeffs = np.array(coeffs) 
        b = np.array(b)
        
        sol = np.linalg.pinv(coeffs) @ b 

        self.open_lst = []
        self.mine_lst = []
        for i, x in enumerate(sol):
            if x < 0.:  # say, likely to an opening
                self.open_lst.append(inv_mapping[i]) 
            elif x > .7:  # say, likely to be a mine 
                self.mine_lst.append(inv_mapping[i])
        
        
        


