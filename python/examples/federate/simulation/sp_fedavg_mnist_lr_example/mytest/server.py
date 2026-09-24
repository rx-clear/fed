import torch
import time

class Server:


    def __init__(self,clients):

        self.clients=clients

        self.model={
            "w":
            torch.zeros(10)
        }


    def aggregate(self,updates):

        print(
            "[Server] Aggregating updates..."
        )
        result={}


        for key in updates[0]:

            result[key]=sum(
                u[key]
                for u in updates
            )/len(updates)


        return result



    def train(self,rounds):


        for r in range(rounds):
            print("\n================")
            print(
                f"Communication Round {r}"
            )
            print("================")
            start=time.time()
            updates=[]


            for c in self.clients:

                update=c.train(
                    self.model
                )

                updates.append(update)



            self.model=self.aggregate(
                updates
            )


            end=time.time()


            print(
                "[Server] "
                "Round finished"
            )


            print(
                f"[Server] "
                f"Time={end-start:.3f}s"
            )


            print(
                "[Server] "
                "Global model:",
                self.model
            )