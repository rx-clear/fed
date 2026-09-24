from client import Client
from server import Server


client_num = 3


clients=[]


for i in range(client_num):

    clients.append(
        Client(i)
    )


server=Server(clients)


server.train(rounds=5)