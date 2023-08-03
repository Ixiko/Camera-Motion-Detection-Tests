import sys
import socket

if sys.platform == "win32":
 	import libs.win32_system_infos as sysinfo

elif sys.platform == "darwin":
    import libs.macOS_system_infos as sysinfo

elif sys.platform == "linux":
    import libs.linux_system_infos as sysinfo

else:
    raise NotImplementedError("Operating system is not supported")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get the IP address of the client
def get_local_ip():
    # Create a temporary socket to get the local IP address.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Connect the socket to a remote host (Google DNS is used here).
        s.connect(("8.8.8.8", 80))

        # Get the IP address of the local host.
        local_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Fehler beim Abrufen der IP-Adresse: {e}")
        local_ip = None
    finally:
        s.close()

    return local_ip


