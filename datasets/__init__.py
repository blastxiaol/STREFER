from .strefer import STREFER

def create_dataset(args, split):
    if args.dataset == "strefer":
        dataset = STREFER(args, split)
    else:
        raise NotImplementedError
    return dataset