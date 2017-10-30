import numpy

def to_str(iteratable):
    return ", ".join([str(e) for e in iteratable])

def file_log(*args):
    msg = args[1:]
    path = args[0]
    content = ""
    if type(msg) == dict:
        for k in msg:
            content += ("{},{}\n".format(k, msg[k]))
    elif type(msg) == list:
        content += to_str(msg) + "\n"
    elif type(msg) == tuple:
        content += to_str(msg) + "\n"
    elif type(msg) == numpy.ndarray:
        content += to_str(msg) + "\n"
    with open(path, "a") as f:
        f.write(content)
    print(content, end="")

if __name__ == '__main__':
    file_log("t.log", "asd", "asd", "er")