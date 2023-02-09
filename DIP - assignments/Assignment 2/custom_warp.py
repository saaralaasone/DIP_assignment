def warp2d(image, inverse_map, ubound=[-1,1], vbound=[-1,1], xbound=[-1,1], ybound=[-1,1], output_shape=None, **kwargs):
    import skimage.transform
    import numpy as np

    def remap(coords, bound_in, bound_out=[[-1,1],[-1,1]]):   
        bound_in, bound_out = np.array(bound_in).T, np.array(bound_out).T        
        # Normalize input coordinates to unit square:
        coords = (coords-bound_in[0]) / (bound_in[1]-bound_in[0])    
        # Map them to the box defined by bound_out:
        coords = coords * (bound_out[1]-bound_out[0]) + bound_out[0]    
        return coords

    input_shape = image.shape
    output_shape = image.shape if not output_shape else output_shape

    output_bound = [xbound, ybound]
    output_bound_in_pixels = [[0,output_shape[1]-1], [0,output_shape[0]-1]]

    input_bound = [ubound, vbound]
    input_bound_in_pixels = [0, input_shape[1]-1], [0,input_shape[0]-1]
    
    inverse_tau = lambda X,**p: remap(inverse_map(remap(X,output_bound_in_pixels, output_bound),**p), input_bound, input_bound_in_pixels)

    warped = skimage.transform.warp(image, inverse_tau, output_shape=output_shape, **kwargs)
    return warped