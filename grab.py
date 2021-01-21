import pynput.keyboard
import pynput.mouse
import platform 
from ctypes import windll
"""
def makeChromeFront():
    
    import win32process 
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
"""      

def getBoxCoordinates():
    """
    get the coordinates of box corner. 


    """

    print('Prepare to get coordinates of box. Press \'r\' when you are ready, and recording will automatically begin. ')
    print('Once recording begins, click on the top-left corner and bottom-right in turn. ')

    def on_release(key):
        try: 
            return not key.char == 'r' 
        except AttributeError:
            return True

    with pynput.keyboard.Listener(on_release=on_release) as listener:
        listener.join()
    print('Recording...')

    def GetCursorPosOnLeftClick():
        with pynput.mouse.Listener(on_click=lambda *_: _[-1]) as listener:
            listener.join()
        return pynput.mouse.Controller().position

    left, top = GetCursorPosOnLeftClick()
    right, bottom = GetCursorPosOnLeftClick() 

    # due to dpi virtualization, the position return by win32gui.getCursorPos is not correct. 
    # check out https://stackoverflow.com/questions/32541475/win32api-is-not-giving-the-correct-coordinates-with-getcursorpos-in-python    

    scale = windll.shcore.GetScaleFactorForDevice(0) / 100 if platform.system() == 'Windows' else 1

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
    # makeChromeFront()
    print(getBoxCoordinates())

