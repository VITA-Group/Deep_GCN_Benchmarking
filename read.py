import os

with open('time_and_memory.log', 'r') as read_file:
    with open('tm.log', 'w') as write_file:
        lines = read_file.readlines()
        for line in lines:
            if '--->' in line:
                write_file.writelines(line)
            else:
                words = line.split(' ')
                write_file.writelines(words[1][:-1]+'/'+words[-1]+'\n')
