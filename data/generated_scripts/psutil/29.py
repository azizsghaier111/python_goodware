import os
import sys
import datetime
import psutil


def print_all_process_info():
    for proc in psutil.process_iter(['pid', 'name', 'exe', 'cwd', 'status', 'create_time']):
        print(f'ID: {proc.info["pid"]}, Name: {proc.info["name"]}, Exe: {proc.info["exe"]}, CWD: {proc.info["cwd"]}, \
Status: {proc.info["status"]}, Created: {datetime.datetime.fromtimestamp(proc.info["create_time"]).isoformat()}')
        try:
            print(f'CPU Affinity: {proc.cpu_affinity()}')
        except psutil.AccessDenied:
            print('CPU Affinity: Access Denied')


def print_all_user_info():
    users = psutil.users()
    for u in users:
        print(f'Name: {u.name}, Terminal: {u.terminal}, Host: {u.host}, \
Started: {datetime.datetime.fromtimestamp(u.started).isoformat()}')


def print_system_boot_info():
    print(f'System Boot Time: {datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()}')


def print_current_process_info():
    pid = os.getpid()
    p = psutil.Process(pid)
    print(f'Current Process ID: {p.pid}, Name: {p.name()}, Exe: {p.exe()}, CWD: {p.cwd()}, \
Status: {p.status()}, Created: {datetime.datetime.fromtimestamp(p.create_time()).isoformat()}, \
CPU Affinity: {p.cpu_affinity()}')


def print_memory_info():
    mem_info = psutil.virtual_memory()
    print(f'Total: {mem_info.total}, Available: {mem_info.available}, Used: {mem_info.used}, Percentage: {mem_info.percent}')


def print_sensors_info():
    try:
        temp_sensors = psutil.sensors_temperatures()
        for name, entries in temp_sensors.items():
            for entry in entries:
                print(f'Name: {entry.label or name}, Current Temp: {entry.current}, High: {entry.high}, Critical: {entry.critical}')
    except AttributeError as e:
        print('No temperature sensors found.')


if __name__ == '__main__':
    for i in range(5):
        print_all_process_info()
        print_all_user_info()
        print_system_boot_info()
        print_current_process_info()
        print_memory_info()
        print_sensors_info()