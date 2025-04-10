import zmq

outpin = 18

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://192.168.1.216:5555")

    message = "GOAL"
    
    running = True
    for i in range(10):
        print("Goal")
        socket.send_string(message)
        
        
    socket.close()
    context.term()
        


    