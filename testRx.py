import zmq

if __name__ == "__main__":
    import zmq

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")

message = socket.recv_string()
print(f"Received message: {message}")

socket.close()
context.term()