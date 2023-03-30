import server

if __name__ == "__main__":
    ser = server.Server()
    ser.init_module()
    ser.train(50, 0.8, 0.5, 4)
    ser.test()