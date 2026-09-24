import torch


class Client:


    def __init__(self,id):

        self.id=id


    def train(self,global_model):

        # 模拟本地训练
        print(
            f"[Client {self.id}] Start local training"
        )
        update = {}

        for k,v in global_model.items():

            update[k]=torch.randn_like(v)
        
        loss=torch.rand(1).item()


        print(
            f"[Client {self.id}] "
            f"Finish training "
            f"loss={loss:.4f}"
        )

        return update