import socket
import numpy as np
import argparse


class opti_client():
    """
    OptiTrack Client: Connects to the server and receives streaming data
    INPUTS:
        server_name:    name/ ip of the server
        port:           port for connection
        packet_size:    Bytes of data being exchanged 
    """
    def __init__(self, server_name, port, packet_size):
        self.server_name = server_name
        self.port = port
        self.packet_size = packet_size
        
        self.client_socket = None
        self.data_raw = None
        self.data_float = None

    def connect(self):
        self.client_socket = socket.socket()  # instantiate
        print("Connecting to: {}:{}...".format(self.server_name, self.port))
        self.client_socket.connect((self.server_name, self.port))  # connect to the server
        print("Connected to: {}:{}".format(self.server_name, self.port))

    def read(self):
        data = self.client_socket.recv(self.packet_size)  # receive response

        if len(data)!=self.packet_size: # if not all data read. Read leftovers
            n_read = len(data)
            n_left = self.packet_size - n_read
            data += self.client_socket.recv(n_left)  # purge unread data
            print("Partial packet of length {} found at t={:3.3f}. Reading additional {} bytes".format(n_read, self.data_float[0], n_left))

        # handle data
        self.data_raw = data
        self.data_float = np.frombuffer(data, dtype=np.float32)
        return self.data_float.copy()

    def isconnected(self):
        return False if self.client_socket is None else True 

    def close(self):
        if self.isconnected():
            self.client_socket.close()  # close the connection
            self.client_socket = None
            print("Disconnected from: {}:{}".format(self.server_name, self.port))
            
    def __del__(self):
        self.close()

# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="OptiTrack Client: Connects to the server and fetches streaming data")

    parser.add_argument("-s", "--server_name", 
                        type=str, 
                        help="IP address or hostname of the server to connect to",
                        default="hulk.cs.washington.edu")
    parser.add_argument("-p", "--port", 
                        type=int, 
                        help="Port to use (> 1024)",
                        default=5000)
    parser.add_argument("-n", "--nbytes",
                        type=int, 
                        help="Size of packets being exchanged",
                        default=36) # [t, id, x, y, z, q0, q1, q2, q3, q4]
    parser.add_argument("-v", "--verbose",
                        type=bool, 
                        help="print data stream",
                        default=False)
    return parser.parse_args()


if __name__ == '__main__':
    
    # get args
    args = get_args()
    
    # Connect and receive streaming data
    oc = opti_client(args.server_name, args.port, args.nbytes)
    oc.connect()
    print("Reading data.... (Press CTRL+C to exit)")
    while oc.isconnected():
        data = oc.read()
        if args.verbose:
            # print(data[1:5])
            print("T:{:03.3f}, id:{:2d}, x:{:+03.3f}, y:{:+03.3f}, z:{:+03.3f}"\
                .format(data[0], int(data[1]), data[2], data[3], data[4]))
    oc.close()