
def define_Model(opt):
    model = opt['model']      # one input: L
    print(model)
    if model == 'plain': #this one
        from models.model_plain import ModelPlain as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
