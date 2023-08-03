import Quartz

# return None only function missing its code, because its not written

def get_monitor_info():
	return None


def is_multi_monitor():
	return None


def get_dpi():
	return None


def get_window_positions():
    def get_window_info(window):
        attributes = window.windowAttributes()
        position = attributes["kCGWindowBounds"]['X'], attributes["kCGWindowBounds"]['Y']
        size = attributes["kCGWindowBounds"]['Width'], attributes["kCGWindowBounds"]['Height']
        return position, size

    window_positions = []
    windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)

    for window in windows:
        position, size = get_window_info(window)
        window_info = {
            "name": window["kCGWindowName"],
            "position": position,
            "size": size
        }
        window_positions.append(window_info)

    return window_positions
