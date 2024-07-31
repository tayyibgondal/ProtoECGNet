def log_to_file_and_console(message, logfile='results.txt'):
    print(message)
    with open(logfile, 'a') as f:
        f.write(message + '\n')

log_to_file_and_console("Test message")
