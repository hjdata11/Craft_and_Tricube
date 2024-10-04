
def Model_factory(backbone, num_classes):
 
    if backbone == 'hourglass104_MRCB_cascade':
        from models.basenet.hourglass_MRCB_cascade import StackedHourGlass as Model
        model = Model(num_classes, 2)
    elif backbone == 'craft':
        from models.basenet.craft import CRAFT as Model
        model = Model()
        
    else:
        raise "Model import Error !! "
        
    return model
