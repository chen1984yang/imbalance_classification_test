
def file_log(*args):
    msg = args[1:]
    path = args[0]
    with open(path, "a") as f:
        if type(msg) == dict:
            for k in msg:
                f.write("{},{}\n".format(k, msg[k]))
        elif type(msg) == list:
            f.write(", ".join([str(e) for e in msg]) + "\n")
        elif type(msg) == tuple:
            f.write(", ".join([str(e) for e in msg]) + "\n")
        print(msg)

if __name__ == '__main__':
    file_log("t.log", "asd", "asd", "er")