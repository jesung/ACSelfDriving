import socket


class ACSocket:
    """
    Connects to AI driver app within Assetto corsa via socket.

    The AC app will connect to host 127.0.0.1 and port 654321 which are the default values of the class.
    Reference code from https://realpython.com/python-sockets/

    Attributes:
        sock (Socket): socket object to make the initial connection
        conn (Socket): socket object usable to send and receive data on the connection
        addr (str): address bound to the socket on the other end of the connection.
        data (str): store data received from the socket connection as str
    """

    sock = None
    conn = None
    addr = None
    data = None

    def __init__(self, host: str = "127.0.0.1", port: int = 65431) -> None:
        """
        Set up socket initial parameters and listen.

        Parameters:
            host (str): Standard loopback interface address (localhost)
            port (int): Port to listen on (non-privileged ports are > 1023)
        """

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()

    def connect(self) -> socket:
        """Accept the connection on target host & port."""
        self.conn, self.addr = self.sock.accept()
        print("Connected by", self.addr)
        return self.conn

    def update(self) -> None:
        """Receive and store data through socket connection."""
        try:
            self.data = self.conn.recv(1024)
            # print(f"Received: {self.data}")
            # self.conn.sendall(self.data)
        except:
            print("Didn't receive data")
            self.on_close()

    def on_close(self) -> None:
        """Ensure socket is properly closed before terminating program."""
        print("Closing socket connection.")
        self.sock.close()
