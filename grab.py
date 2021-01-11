import PIL.ImageGrab
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import time
# for i in tqdm(range(1000)):

# left, top, right, bottom = bbox

"""
pix = np.array(PIL.ImageGrab.grab(bbox = (796, 239, 2628, 1629)))
plt.imshow(pix)
plt.show()
""" 

import win32gui 
import win32api
import win32process 

def makeChromeFront():
    def windowEnumerationHandler(hwnd, unused):
        nonlocal number
        if 'chrome' in win32gui.GetWindowText(hwnd).lower():
            number = hwnd
            
    
    number = -1
    win32gui.EnumWindows(windowEnumerationHandler,[]) 

    if number == -1:
        raise RuntimeError
    win32gui.ShowWindow(number,5)
    win32gui.SetForegroundWindow(number)
        

def getBoxCoordinates(scale=2.5):
    """
    get the coordinates of box corner. 

    scale: due to dpi virtualization, the position return by win32gui.getCursorPos is not correct. 

    check out https://stackoverflow.com/questions/32541475/win32api-is-not-giving-the-correct-coordinates-with-getcursorpos-in-python
    """

    print('Prepare to get coordinates of box. Press \'r\' when you are ready, and recording will automatically begin. ')
    print('Once recording begins, click on the top-left corner and bottom-right in turn. ')

    while True:
        if win32api.GetAsyncKeyState(ord('R')):
            print('Recording... ')
            break 

    def GetCursorPosOnLeftClick():
        state_left = win32api.GetKeyState(0x01)  # Left button down = 0 or 1. Button up = -127 or -128
        state_right = win32api.GetKeyState(0x02)  # Right button down = 0 or 1. Button up = -127 or -128

        while True:
            a = win32api.GetKeyState(0x01)
            b = win32api.GetKeyState(0x02)
            if a != state_left:  # Button state changed
                state_left = a
                if a < 0:
                    return win32gui.GetCursorPos()

            if b != state_right:  # Button state changed
                state_right = b
            
            time.sleep(0.001)
    
    left, top = GetCursorPosOnLeftClick()
    
    right, bottom = GetCursorPosOnLeftClick() 
    
    return left*scale, top*scale, right*scale, bottom*scale

def color_mask(array, r,g,b):
    """
    array : m x n x 3 array of colors
    *_lim are 2-element tuples, where the first element is expected to be <= the second.
    """
    r_mask = (array[..., 0] == r) 
    g_mask = (array[..., 1] == g)
    b_mask = (array[..., 2] == b)
    
    return r_mask & g_mask & b_mask

def equal_mask(array):
    """
    array : m x n x 3 array of colors
    """
    return (array[..., 0] == array[..., 1])  & (array[..., 0] == array[..., 2])

if __name__ == '__main__':
    makeChromeFront()
    print(getBoxCoordinates())

