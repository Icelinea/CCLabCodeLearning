import datetime

# open
def open_log_file(file_name = None):
    file = open('./log_results/' + file_name, 'w', encoding='utf-8')
    return file

# close
def close_log_file(file = None):
    file.close()

# print
def log(msg = '', file = None, print_msg = True, end = '\n'):
    """
    msg 表示打印信息, file 为日志文件对象,
    print_msg 表示是否在控制台打印 msg 信息,
    end 为写入文件默认结尾
    """
    if print_msg:
        print(msg)
    
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)
    
    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]

    for line in lines:
        if line == lines[-1]:
            file.write('[' + t + ']' + str(line) + end)
        else:
            file.write('[' + t + ']' + str(line))