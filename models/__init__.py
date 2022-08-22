from .vil_bert3d import ViLBert3D
from .vil_bert3d_ff import ViLBert3DFF

def create_model(args):
    if args.feature_fusion:
        model = ViLBert3DFF(args)
    else:
        model = ViLBert3D(args)
    return model