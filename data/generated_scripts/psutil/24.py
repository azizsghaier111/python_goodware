import os
import sys
import datetime
import psutil 


def print_users():
    print("\n# Users Info")
    users = psutil.users()
    for user in users:
        print(f'User: {user.name}, Terminal: {user.terminal}, Host: {user.host}, Started: {datetime.datetime.fromtimestamp(user.started).isoformat()}')


def print_boot_time():
    print("\n# Boot time")
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
    print(f'Boot Time: {boot_time.isoformat()}')


def print_processes():
    print("\n# Running processes Info")
    for proc in psutil.process_iter(['name', 'exe', 'cwd', 'status', 'create_time']):
        print(f'ID: {proc.info["exe"]}, Process Name: {proc.info["name"]}, Exe: {proc.info["exe"]}, CWD: {proc.info["cwd"]}, Status: {proc.info["status"]}, Created: {datetime.datetime.fromtimestamp(proc.info["create_time"]).isoformat()}')
        

def get_process_info(process_id):
    try:
        p = psutil.Process(process_id)
        details = {
                    "name": p.name(),
                    "exe": p.exe(),
                    "cwd": p.cwd(),
                    "status": p.status(),
                    "create_time": datetime.datetime.fromtimestamp(p.create_time()).isoformat(),
                    "cpu_times": p.cpu_times(),
                    "cpu_affinity": p.cpu_affinity(),
                    "is_running": p.is_running(),
                  }
        return details
    except psutil.NoSuchProcess:
        return None


def main():
    print_processes()
    print_users()
    print_boot_time()
    
    current_pid = os.getpid()
    print(get_process_info(current_pid))


if __name__ == "__main__":
    main()