import sys, psutil, os, signal

######################################### WIPES RESOURCES #########################################

# if a simulation needs to be killed this program is useful
def kill_processes(func):
    
    for process in psutil.process_iter():
        if process.cmdline() == ['python3', func]:
            print('Process found. Terminating it.')
            os.kill(process.pid, signal.SIGKILL)
            break
        
if __name__ == "__main__":
    for _ in range(os.cpu_count()):
        kill_processes(sys.argv[1])

#:$ python3 clear.py func.py
