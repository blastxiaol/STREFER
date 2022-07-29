from .strefer import STREFER
from .strefer_mf import STREFER_MF

def create_dataset(args, split):
    if args.dataset == "strefer" and not args.multi_frame:
        dataset = STREFER(args, split)
    elif args.dataset == "strefer" and args.multi_frame:
        dataset = STREFER_MF(args, split)
    else:
        raise NotImplementedError
    return dataset