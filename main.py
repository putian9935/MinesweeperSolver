import grab
import PIL.ImageGrab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import win32api 


class Solver:
    colors = {'line': (128, 128, 128), 'unopened':(198,198,198), 'background': (196, 196, 196), 1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 
    4: (0, 0, 128), 5: (128, 0, 0), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128)}  # colors corresponding to different number

    def __init__(self,):
        # 
        self.bbox = grab.getBoxCoordinates()
        # print(self.bbox)
        # grab.makeChromeFront()
        # self.bbox = (797.5, 375.0, 2675.0, 1642.5)
        
        # self._takeSnapshot(fromClipboard=True)
        self._takeSnapshot()
        self._showSnapshot()
        self._getDimension(Solver._filterColor(self.pix, 'line'))
        
        # self._reveal_map()
        print('Horizontal dimension is: %d; vertical dimension is: %d' %
              (self.horz_dim, self.vert_dim))

        self.h_size, self.v_size = round((np.max(self.horz_cent) - np.min(self.horz_cent)) / self.horz_dim), round(
            (np.max(self.vert_cent) - np.min(self.vert_cent)) / self.vert_dim)
        print('The map size is: %dx%d' % (self.h_size, self.v_size))
        
        self.state = -2*np.ones((self.v_size, self.h_size))  # initially, all states are unknown
        self.frontiers = []

        self.openAscending = 1


    def solve(self):
        
        import pyautogui
        dirs = [(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(-1,0),(1,0)]
        def _accumulateMines(i, j):
            ret = 0 
            for di, dj in dirs:
                if 0 <= i+di < self.v_size and 0 <= j + dj < self.h_size:
                    if self.state[i+di, j+dj] == -1:  # -1 is mine 
                        ret += 1 
            return ret 

        def _accumulateUnopened(i,j):
            ret = 0 
            for di, dj in dirs:
                if 0 <= i+di < self.v_size and 0 <= j + dj < self.h_size:
                    if self.state[i+di, j+dj] == -2:  # -2 is unopened 
                        ret += 1 
            return ret 

        def _appendToBeOpenedList(i, j):
            for di, dj in dirs:
                if 0 <= i+di < self.v_size and 0 <= j + dj < self.h_size:
                    if self.state[i+di, j+dj] == -2:  # -2 is unopened 
                        to_be_opened.add((i+di, j+dj))

        def _labelUnopenedAsMine(i, j):
            for di, dj in dirs:
                if 0 <= i+di < self.v_size and 0 <= j + dj < self.h_size:
                    if self.state[i+di, j+dj] == -2:  # -2 is unopened 
                        self.state[i+di, j+dj] = -1

        def _open(coord_lst, scale=2.5):
            coord_lst = sorted(list(coord_lst), key=lambda _: self.openAscending * _[1])
            self.openAscending *= -1
            for i, j in reversed(coord_lst):
                # bbox[0] is horizontal, bbox[1] is vertical
                if self.state[i, j] != -2:
                    continue
                real_x = self.bbox[0] + self.horz_cent[j] + self.horz_dim // 2 + self.horz_dim /5 * (np.random.rand()-.5)
                real_y = self.bbox[1] + self.vert_cent[i] + self.vert_dim // 2  + self.vert_dim /5 * (np.random.rand()-.5)

                
                pyautogui.moveTo((int(real_x), int(real_y)),  pause=0.0)
                pyautogui.click()
                # self._takeSnapshot()
                # self.getState()
                # win32api.Sleep(100+int(100*(np.random.rand()-.5)))
            
        def _formAdjacentPairs():
            def isAdjacent(pt1, pt2):
                return (abs(pt1[0] - pt2[0]) ==1 and pt1[1]==pt2[1]) or (abs(pt1[1] - pt2[1]) ==1 and pt1[0]==pt2[0])

            ret = []
            for i in range(len(self.frontiers)):
                for j in range(i + 1, len(self.frontiers)):
                    if isAdjacent(self.frontiers[i],  self.frontiers[j]):
                        ret.append((self.frontiers[i], self.frontiers[j]))  # returned are coordinates

            return ret 

        def _checkMine(i, j):
            if 0 <= i< self.v_size and 0 <= j< self.h_size:
                if self.state[i,j] == -1:
                    return True 
            return False

        def _checkUnopened(i, j):
            if 0 <= i< self.v_size and 0 <= j< self.h_size:
                if self.state[i,j] == -2:
                    return True 
            return False 

        def _appendToBeOpenedListF(i, j):
            if _checkUnopened(i, j):
                to_be_opened.add((i, j))

        def _labelUnopenedAsMineF(i, j):
            if _checkUnopened(i, j):
                self.state[i, j] = -1
                

        def _useFormula():
            adjacent_pairs = _formAdjacentPairs()
            
            for p1, p2 in adjacent_pairs:
                if self.state[p1] > self.state[p2]:
                    p1, p2 = p2, p1  # p2 side has more mines
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1] 
                if dx == 0:
                    m1 = _checkMine(p1[0]-1, p1[1] - dy) + _checkMine(p1[0], p1[1] - dy) + _checkMine(p1[0]+1, p1[1] - dy)
                    m2 = _checkMine(p2[0]-1, p2[1] + dy) + _checkMine(p2[0], p2[1] + dy) + _checkMine(p2[0]+1, p2[1] + dy)
                    u1 = _checkUnopened(p1[0]-1, p1[1] - dy) + _checkUnopened(p1[0], p1[1] - dy) + _checkUnopened(p1[0]+1, p1[1] - dy)
                    u2 = _checkUnopened(p2[0]-1, p2[1] + dy) + _checkUnopened(p2[0], p2[1] + dy) + _checkUnopened(p2[0]+1, p2[1] + dy)

                    if m1 + self.state[p2] - self.state[p1] == m2 + u2: 
                        # print(u1, u2)
                        # self._revealToBeOpenedList([p1, p2])
                        # sself._revealToBeOpenedList([(p1[0]-1, p1[1] - dy), (p1[0], p1[1] - dy), (p1[0]+1, p1[1] - dy)])
                        if u2:
                            _labelUnopenedAsMineF(p2[0]-1, p2[1] + dy) 
                            _labelUnopenedAsMineF(p2[0], p2[1] + dy) 
                            _labelUnopenedAsMineF(p2[0]+1, p2[1] + dy) 
                        if u1: 
                            _appendToBeOpenedListF(p1[0]-1, p1[1] - dy)
                            _appendToBeOpenedListF(p1[0], p1[1] - dy)
                            _appendToBeOpenedListF(p1[0]+1, p1[1] - dy)  
                else:
                    m1 = _checkMine(p1[0]-dx, p1[1] - 1) + _checkMine(p1[0]-dx, p1[1]) + _checkMine(p1[0]-dx, p1[1] + 1)
                    m2 = _checkMine(p2[0]+dx, p2[1] - 1) + _checkMine(p2[0]+dx, p2[1]) + _checkMine(p2[0]+dx, p2[1] + 1)
                    u1 = _checkUnopened(p1[0]-dx, p1[1] - 1) + _checkUnopened(p1[0]-dx, p1[1]) + _checkUnopened(p1[0]-dx, p1[1] + 1)
                    u2 = _checkUnopened(p2[0]+dx, p2[1] - 1) + _checkUnopened(p2[0]+dx, p2[1]) + _checkUnopened(p2[0]+dx, p2[1]  + 1)

                    if m1 + self.state[p2] - self.state[p1] == u2+ m2: 
                        # print(u1, u2)
                        # self._revealToBeOpenedList([p1, p2])
                        
                        if u2:
                            _labelUnopenedAsMineF(p2[0]+dx, p2[1] - 1)
                            _labelUnopenedAsMineF(p2[0]+dx, p2[1]) 
                            _labelUnopenedAsMineF(p2[0]+dx, p2[1]  + 1) 
                        if u1: 
                            _appendToBeOpenedListF(p1[0]-dx, p1[1] - 1)
                            _appendToBeOpenedListF(p1[0]-dx, p1[1])
                            _appendToBeOpenedListF(p1[0]-dx, p1[1] + 1)

                        
                if self.state[p1] == self.state[p2]:  # do it again 
                    p1, p2 = p2, p1
                    
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1] 
                if dx == 0:
                    m1 = _checkMine(p1[0]-1, p1[1] - dy) + _checkMine(p1[0], p1[1] - dy) + _checkMine(p1[0]+1, p1[1] - dy)
                    m2 = _checkMine(p2[0]-1, p2[1] + dy) + _checkMine(p2[0], p2[1] + dy) + _checkMine(p2[0]+1, p2[1] + dy)
                    u1 = _checkUnopened(p1[0]-1, p1[1] - dy) + _checkUnopened(p1[0], p1[1] - dy) + _checkUnopened(p1[0]+1, p1[1] - dy)
                    u2 = _checkUnopened(p2[0]-1, p2[1] + dy) + _checkUnopened(p2[0], p2[1] + dy) + _checkUnopened(p2[0]+1, p2[1] + dy)

                    if m1 + self.state[p2] - self.state[p1] == m2 + u2: 
                        # print(u1, u2)
                        # self._revealToBeOpenedList([p1, p2])
                        # sself._revealToBeOpenedList([(p1[0]-1, p1[1] - dy), (p1[0], p1[1] - dy), (p1[0]+1, p1[1] - dy)])
                        if u2:
                            _labelUnopenedAsMineF(p2[0]-1, p2[1] + dy) 
                            _labelUnopenedAsMineF(p2[0], p2[1] + dy) 
                            _labelUnopenedAsMineF(p2[0]+1, p2[1] + dy) 
                        if u1: 
                            _appendToBeOpenedListF(p1[0]-1, p1[1] - dy)
                            _appendToBeOpenedListF(p1[0], p1[1] - dy)
                            _appendToBeOpenedListF(p1[0]+1, p1[1] - dy)  
                else:
                    m1 = _checkMine(p1[0]-dx, p1[1] - 1) + _checkMine(p1[0]-dx, p1[1]) + _checkMine(p1[0]-dx, p1[1] + 1)
                    m2 = _checkMine(p2[0]+dx, p2[1] - 1) + _checkMine(p2[0]+dx, p2[1]) + _checkMine(p2[0]+dx, p2[1] + 1)
                    u1 = _checkUnopened(p1[0]-dx, p1[1] - 1) + _checkUnopened(p1[0]-dx, p1[1]) + _checkUnopened(p1[0]-dx, p1[1] + 1)
                    u2 = _checkUnopened(p2[0]+dx, p2[1] - 1) + _checkUnopened(p2[0]+dx, p2[1]) + _checkUnopened(p2[0]+dx, p2[1]  + 1)

                    if m1 + self.state[p2] - self.state[p1] == u2+ m2: 
                        # print(u1, u2)
                        # self._revealToBeOpenedList([p1, p2])
                        
                        if u2:
                            _labelUnopenedAsMineF(p2[0]+dx, p2[1] - 1)
                            _labelUnopenedAsMineF(p2[0]+dx, p2[1]) 
                            _labelUnopenedAsMineF(p2[0]+dx, p2[1]  + 1) 
                        if u1: 
                            _appendToBeOpenedListF(p1[0]-dx, p1[1] - 1)
                            _appendToBeOpenedListF(p1[0]-dx, p1[1])
                            _appendToBeOpenedListF(p1[0]-dx, p1[1] + 1)
                
        
        
        attempt = 0
        while True:
            self._takeSnapshot()
            self.getState()
            to_be_opened = set()
            tmp_frontiers = []
            
            while self.frontiers:
                i, j = self.frontiers.pop()
                mines = _accumulateMines(i, j) 
                unopened = _accumulateUnopened(i, j) 
                if mines + unopened == self.state[i, j]:  # okay, all around are mines 
                    if unopened: 
                        _labelUnopenedAsMine(i, j)
                    continue
                if mines == self.state[i, j]:  # okay, chording possible
                    if unopened:
                        _appendToBeOpenedList(i, j)
                    continue
                # cannot determine now, save it for next round 
                tmp_frontiers.append((i, j))
                
            while tmp_frontiers:
                self.frontiers.append(tmp_frontiers.pop())

            if not to_be_opened:
                _useFormula()
                if not to_be_opened:
                    attempt += 1
                    print('Auto solve failed at attempt #%d. ' % attempt)
                else:
                    attempt = 0
                if attempt > 5:
                    break
                    self._revealState()

            # self._revealToBeOpenedList(to_be_opened)
            if not self.frontiers:
                break
            _open(to_be_opened)

                    


    def getState(self):
        """
        Assume that _takeSnapshot is called before. 
        """
        for i in range(self.v_size):
            for j in range(self.h_size):
                if self.state[i, j] != -2: continue  # only update when necessary
                hmid, vmid = (self.horz_cent[j] + self.horz_cent[j+1]
                              ) // 2, (self.vert_cent[i] + self.vert_cent[i+1]) // 2
                area = self.pix[vmid-5:vmid+5, hmid-5:hmid+5]
                
                isbreak = False
                for c in range(1, 9):  # check number 
                    if np.any(Solver._filterColor(area, c)):
                        if c == 7 and not np.all(area[...,0]==0): 
                            continue
                            
                        self.state[i, j] = c 
                        self.frontiers.append((i, j))
                        isbreak=True
                        break
                if isbreak:
                        continue 
                
                if np.all(Solver._filterColor(area, 'background')):
                    self.state[i, j] = 0 
                    continue 
                    
                if np.all(Solver._filterColor(area, 'unopened')):
                    continue

                self.state[i, j] = -1
                # if not np.all(Solver._filterColor(area[grab.equal_mask(area)], 'background')):
                # if np.any(Solver._filterColor(area, 'flag')):
                    

    def _takeSnapshot(self, fromClipboard=False):
        if fromClipboard:
            self.pix = np.array(PIL.ImageGrab.grabclipboard())
        else:
            self.pix = np.array(PIL.ImageGrab.grab(bbox=self.bbox))

    def _showSnapshot(self):
        plt.imshow(self.pix)
        plt.show()

    @staticmethod
    def _filterColor(arr, color):
        return grab.color_mask(arr, *Solver.colors[color]).astype(np.int32)

    def _getDimension(self, mask, horz_thres=None, vert_thres=None):
        def _getClusterCentroid(x, crit=10):
            ret = []

            cur_cluster = [x[0]]
            for i in range(len(x)-1):
                if x[i + 1] - x[i] >= crit:  # checkout here
                    ret.append(sum(cur_cluster) / len(cur_cluster))
                    cur_cluster = [x[i+1]]
                else:
                    cur_cluster.append(x[i+1])
            ret.append(sum(cur_cluster) / len(cur_cluster))

            return np.array(ret)

        def _labelClusterCentroid(x):
            diff = np.min(x[1:] - x[:-1])
            ret = []
            for val in x:
                ret.append(round((val-x[0])/diff))
            return np.array(ret)

        def _filterClusterCentroid(x, horz=True, horz_lim_l=50, horz_lim_h=80, vert_lim_l=210, vert_lim_h=150):
            lim_l, lim_h = (horz_lim_l, self.pix.shape[1]-horz_lim_h) if horz else (
                vert_lim_l, self.pix.shape[0]-vert_lim_h)
            return x[(x >= lim_l) & (x <= lim_h)]

        def _insertMissing(x, horz=True, horz_lim_l=50, horz_lim_h=80, vert_lim_l=210, vert_lim_h=150):
            ret = []
            dim, lim_l, lim_h = (self.horz_dim, horz_lim_l, self.pix.shape[1]-horz_lim_h) if horz else (
                self.vert_dim, vert_lim_l, self.pix.shape[0]-vert_lim_h)

            cur = np.min(x)
            while cur > lim_l:
                cur -= dim
                ret.append(cur)

            cur = np.max(x)
            while cur < lim_h:
                cur += dim
                ret.append(cur)

            for i in range(len(x) - 1):
                if round((x[i+1]-x[i])/dim) == 1:
                    continue
                cur = x[i]
                while round((x[i+1]-cur)/dim) > 1:
                    cur += dim
                    ret.append(cur)
            ret.extend(x)
            return np.array(sorted(ret))

        horz = np.sum(mask, axis=0)
        vert = np.sum(mask, axis=1)

        if not horz_thres:
            horz_thres = (.5*np.max(horz), .9 * np.max(horz))
        if not vert_thres:
            vert_thres = (.5*np.max(vert), .95 * np.max(vert))

        horz_range = np.arange(len(horz))
        horz_mask = ((horz >= horz_thres[0]) & (horz <= horz_thres[1]))
        horz_cent = _getClusterCentroid(horz_range[horz_mask])
        horz_cent = _filterClusterCentroid(horz_cent, True)
        self.horz_dim = round(np.min(horz_cent[1:] - horz_cent[:-1]))

        vert_range = np.arange(len(vert))
        vert_mask = ((vert >= vert_thres[0]) & (vert <= vert_thres[1]))
        vert_cent = _getClusterCentroid(vert_range[vert_mask])
        vert_cent = _filterClusterCentroid(vert_cent, False)
        self.vert_dim = round(np.min(vert_cent[1:] - vert_cent[:-1]))

        self.horz_cent = _insertMissing(horz_cent, True).astype(np.int32)
        self.vert_cent = _insertMissing(vert_cent, False).astype(np.int32)


    def _revealToBeOpenedList(self, coord_lst):
        print(coord_lst)
        fig, ax = plt.subplots()
        ax.imshow(self.pix, interpolation=None)
        for i, j in coord_lst:
            rect = patches.Rectangle(( self.horz_cent[j] + self.horz_dim/4,self.vert_cent[i] + self.vert_dim/4,), self.vert_dim/2, self.horz_dim/2, facecolor='r', alpha=.5)
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


    def _revealState(self):
        plt.matshow(self.state)
        plt.show()


if __name__ == '__main__':
    import cProfile, pstats, io
    pr = cProfile.Profile()

    s =  Solver()
    # pr.enable()

    s.solve()

    '''
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(.1)  # only 10 percent is more than enough
    print(s.getvalue())
    '''
