import ctypes
import win32api # pip install pywin32
import pygetwindow as gw

def get_monitor_info():
	monitors     = []
	monitor_info = win32api.EnumDisplayMonitors(None, None)

	try:
		for monitor in monitor_info:
			monitor_dict = {}
			monitor_dict['monitor'] = monitor[0]
			monitor_dict['rect']    = monitor[2]
			monitors.append(monitor_dict)
		return monitors

	except win32api.error:
		return None


def is_multi_monitor():
	monitors     = get_monitor_info()
	if monitors is None:
		return None, None
	num_monitors = len(monitors)
	print(f"number of monitors: {num_monitors}")

	# Check if monitors are arranged side by side
	are_adjacent = all(
    	monitors[i]['rect'].right == monitors[i + 1]['rect'].left
    	for i in range(num_monitors - 1)
	)
	return num_monitors, are_adjacent


def get_dpi():
	this_monitors = []
	shcore   = ctypes.windll.shcore
	monitors = win32api.EnumDisplayMonitors()
	hresult  = shcore.SetProcessDpiAwareness(1)    # Support high DPI displays
	assert hresult == 0
	dpiX = ctypes.c_uint()
	dpiY = ctypes.c_uint()
	for i, monitor in enumerate(monitors):
		shcore.GetDpiForMonitor(monitor[0].handle, 0,	ctypes.byref(dpiX), ctypes.byref(dpiY))
		this_monitors.append(dpiX.value/96)
	return this_monitors


def get_window_positions():
  windows = gw.getWindows()
  window_positions = []

  for window in windows:
    window_info = {
      "name": window.title,
      "position": (window.left, window.top),
      "size": (window.width, window.height)
    }
    window_positions.append(window_info)

  return window_positions
