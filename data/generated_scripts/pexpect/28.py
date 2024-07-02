import pexpect

def spawn_child_program():
    child = pexpect.spawn('python')
    return child

def send_commands(child, commands):
    for command in commands:
        child.sendline(command)
    return child

def interact_with_processes():
    child = spawn_child_program()
    commands = ['print("Hello World")', 'for i in range(10): print(i)']
    child = send_commands(child, commands)
    child.interact()

if __name__ == '__main__':
    try:
        interact_with_processes()
    except Exception as e:
        print('The program encountered an error:', e)