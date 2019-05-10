import cv2
import numpy as np
import win32con
import win32gui
import win32ui

from consts import GAME_WIDTH, GAME_HEIGHT

game_x = 0
game_y = 0


def set_game_demensions(hwnd):
    global game_x, game_y
    rect = win32gui.GetWindowRect(hwnd)
    game_x = rect[0] + 3
    game_y = rect[1] + 26  # get rid of title bar


def find_game_window():
    hwid = win32gui.FindWindow(None, "Grand Theft Auto V")
    if hwid != 0:
        set_game_demensions(hwid)
    else:
        # raise Exception("Cannot find game window!")
        pass


def display_screen(screen):
    cv2.imshow('image', screen)
    cv2.resizeWindow('image', GAME_WIDTH, GAME_HEIGHT)
    cv2.waitKey(delay=1)


def grab_screen():
    hwin = win32gui.GetDesktopWindow()
    find_game_window()

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, GAME_WIDTH, GAME_HEIGHT)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (GAME_WIDTH, GAME_HEIGHT), srcdc, (game_x, game_y), win32con.SRCCOPY)

    signed_ints_array = bmp.GetBitmapBits(True)
    img = np.fromstring(signed_ints_array, dtype='uint8')
    img.shape = (GAME_HEIGHT, GAME_WIDTH, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    colourized = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return colourized
    # return cv2.resize(colourized, (IMAGE_SIZE, IMAGE_SIZE))
