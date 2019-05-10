# Citation: Box Of Hats (https://github.com/Box-Of-Hats )
import string

import win32api as wapi


def grab_keys():
    return [key for key in string.printable if wapi.GetAsyncKeyState(ord(key))]
