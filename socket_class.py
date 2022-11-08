import socket


# from https://realpython.com/python-sockets/
class ACSocket:
    # declare class variables
    sock = None
    conn = None
    addr = None
    data = None

    def __init__(self, host, port):
        # Host: Standard loopback interface address (localhost)
        # Post: Port to listen on (non-privileged ports are > 1023)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()

    def connect(self):
        self.conn, self.addr = self.sock.accept()
        print(f"Connected by {self.addr}")

    def update(self):
        try:
            self.data = self.conn.recv(1024)
            # print(f"Received: {self.data}")
            # self.conn.sendall(self.data)
        except:
            print("Didn't receive data")

    def on_close(self):
        self.sock.close()
