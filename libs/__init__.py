import sys


if sys.platform == "win32":
    #from .win32_system_infos import get_window_positions
    #from .win32_system_infos import get_dpi
    from . import win32_system_infos as sysinfo
 	#import libs.win32_system_infos as sysinfo

elif sys.platform == "darwin":
    import libs.macOS_system_infos as sysinfo

elif sys.platform == "linux":
    import libs.linux_system_infos as sysinfo

else:
    raise NotImplementedError("Operating system is not supported")

from . import system_infos as lan

