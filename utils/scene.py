class Scene:
    def __init__(self, name, image_name, box, frame_num=1):
        self.name = name                # str
        self.image_name = image_name    # str
        self.boxes = box            # List[List]
        self.frame_num = frame_num  # int
        self.feature = []   # List[Tuple] 1024 cpu

        self.prev_scene = None  # Scene
    
    def set_prev(self, prev_scene):
        self.prev_scene = prev_scene
