from .vil_bert3d import ViLBert3D
from .vil_bert3d_mf import ViLBert3DMF
from .vil_bert3d_ff import ViLBert3DFF
from .vil_bert3d_bev import ViLBert3DBEV

def create_model(args):
    if args.feature_fusion:
        model = ViLBert3DFF(args)
    elif args.use_bev:
        model = ViLBert3DBEV(args)
    else:
        model = ViLBert3D(args)
    return model