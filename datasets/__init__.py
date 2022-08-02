from .strefer import STREFER
from .strefer_mf import STREFER_MF
from .strefer_mfgt import STREFER_MFGT
from .strefer_ff import STREFER_FF

def create_dataset(args, split):
    if args.dataset == "strefer" and not args.multi_frame:
        dataset = STREFER(args, split)
    elif args.dataset == "strefer" and args.multi_frame:
        dataset = STREFER_MF(args, split)
    elif args.dataset == "strefer_gt":
        dataset = STREFER_MFGT(args, split)
    elif args.dataset == "strefer" and args.feature_fusion:
        dataset = STREFER_FF(args, split)
    else:
        raise NotImplementedError
    return dataset