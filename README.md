## Custom Layers

- cl_lognorm

    scale * np.exp(probability-np.mean(probability))+\
    np.square(probability-np.mean(probability)) + \
    np.min(probability) * np.log(np.square(probability-np.mean(probability)))

- cl_pnorm

    dst_data = np.mean(np.round(layer,6) + np.round(layer,5) + np.round(layer,4) + np.round(layer,3) + np.round(layer,2))
