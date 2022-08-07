from .strefer import STREFER
from .strefer_mf import STREFER_MF
from .strefer_ff import STREFER_FF
from .strefer_bev import STREFER_BEV

def create_dataset(args, split):
    if args.dataset == "strefer":
        if args.feature_fusion:
            dataset = STREFER_FF(args, split)
        elif args.use_bev:
            dataset = STREFER_BEV(args, split)
        else:
            dataset = STREFER_MF(args, split)
    else:
        raise NotImplementedError
    return dataset
