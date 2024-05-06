import socket

def server_listen(bind_ip, bind_port, allowed_ip):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((bind_ip, bind_port))
        server_socket.listen(1)
        print("Server is listening for connections...")

        while True:
            # Accept a client connection
            conn, addr = server_socket.accept()
            print(f"Connection attempted from {addr[0]}")

            # Check if the connection is from the allowed IP
            if addr[0] == allowed_ip:
                print(f"Connected by {addr}")
                with conn:
                    with open('received_output.wav', 'wb') as f:
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                break
                            f.write(data)

                    print("File received successfully.")

                    # Send a confirmation message back to the client
                    conn.sendall(b"File received successfully.")
            else:
                print(f"Connection rejected from {addr[0]}")
                conn.close()
    
if __name__ == "__main__":
    LOCAL_IP = '0.0.0.0'  # Listen on all network interfaces
    PORT = 50007
    # REMOTE_IP = '192.168.1.154'
    REMOTE_IP = "172.26.128.166"
    server_listen(LOCAL_IP, PORT, REMOTE_IP)