"""Wait for all Linux process exits, including non-child processes."""
import os, selectors, sys
def wait_all(pids):
    with selectors.DefaultSelector() as selector:
        try:
            for pid in set(map(int,pids)):
                try: fd=os.pidfd_open(pid)
                except ProcessLookupError: continue
                selector.register(fd,selectors.EVENT_READ)
            while selector.get_map():
                for key,_ in selector.select():
                    selector.unregister(key.fd)
                    os.close(key.fd)
        finally:
            for key in list(selector.get_map().values()):
                selector.unregister(key.fd)
                os.close(key.fd)
if __name__=="__main__": wait_all(sys.argv[1:])
