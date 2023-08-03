from Xlib import display

# return None only function missing its code, because its not written

def get_monitor_info():
	return None


def is_multi_monitor():
	return None


def get_dpi():
	return None


def get_window_positions():
    def get_window_info(window):
        geometry = window.get_geometry()
        position = geometry.x, geometry.y
        size = geometry.width, geometry.height
        return position, size

    window_positions = []
    display_obj = display.Display()
    root = display_obj.screen().root
    window_ids = root.query_tree().children

    for window_id in window_ids:
        try:
            window = display_obj.create_resource_object('window', window_id)
            position, size = get_window_info(window)
            window_info = {
                "name": "Unbekannt",  # Fenstername auf Linux nicht einfach auslesbar
                "position": position,
                "size": size
            }
            window_positions.append(window_info)
        except display.Xlib.error.XError:
            pass

    return window_positions
