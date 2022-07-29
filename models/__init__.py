from .vil_bert3d import ViLBert3D

def create_model(args):
    model = ViLBert3D(args)
    return model