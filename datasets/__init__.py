from .strefer import STREFER
from .strefer_mf import STREFER_MF
from .strefer_mfgt import STREFER_MFGT

def create_dataset(args, split):
    if args.dataset == "strefer" and not args.multi_frame:
        dataset = STREFER(args, split)
    elif args.dataset == "strefer" and args.multi_frame:
        dataset = STREFER_MF(args, split)
    elif args.dataset == "strefer_gt":
        dataset = STREFER_MFGT(args, split)
    else:
        raise NotImplementedError
    return dataset