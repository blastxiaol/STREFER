from .strefer import STREFER
from .strefer_mf import STREFER_MF
from .strefer_ff import STREFER_FF

def create_dataset(args, split):
    if args.dataset == "strefer":
        if args.feature_fusion:
            dataset = STREFER_FF(args, split)
        else:
            dataset = STREFER_MF(args, split)
    else:
        raise NotImplementedError
    return dataset
