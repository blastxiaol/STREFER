from .vil_bert3d import ViLBert3D
from .vil_bert3d_mf import ViLBert3DMF

def create_model(args):
    if args.multi_frame:
        model = ViLBert3DMF(args)
    else:
        model = ViLBert3D(args)
    return model