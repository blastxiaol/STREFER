import os
import os.path as osp
import datetime
import torch

class Logger:
    def __init__(self, work_dir=None) -> None:
        current_date = datetime.datetime.now()
        month = current_date.month
        day = current_date.day
        hour = current_date.hour
        minute = current_date.minute
        sec = current_date.second
        self.work_dir = work_dir
        if self.work_dir is None:
            self.work_dir = f"work_dir/{month}-{day}-{hour}-{minute}-{sec}"
        else:
            self.work_dir = osp.join(self.work_dir, f"{month}-{day}-{hour}-{minute}-{sec}")
        if not osp.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.log = osp.join(self.work_dir, "log.log")

        f = open(self.log, 'w')
        f.close()
    
    def __call__(self, info):
        with open(self.log, 'a') as f:
            info += "\n"
            f.write(info)
    
    def save_model(self, model, path):
        torch.save(model.state_dict(),osp.join(self.work_dir, path))
    
    def load_model(self, model, path):
        return model.load_state_dict(torch.load(path))