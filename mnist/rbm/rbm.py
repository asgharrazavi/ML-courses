def rbm(X,n_inp,n_hid,last_RBM=False):
    """
    inputs:
    X: float, 2d array, shape=(number of samples,n_inp) 
    n_inp: int, number of nodes for the input layer
    n_hid: int, number of nodes for the hidden layer
    """

    print "data.shape:", X.shape
    w = np.random.normal(loc=0.0,scale=0.1,size=(n_inp,n_hid))
    v = np.zeros((1,n_inp))
    h = np.zeros((1,n_hid))
    new_w = np.zeros((n_inp,n_hid))
    new_v = np.zeros((1,n_inp))
    new_h = np.zeros((1,n_hid))
    batchposhidprobs = np.zeros((X.shape[0],n_hid));
    
    for e in range(epochs):
        err_sum = 0
        if e > 5 : p = p_final
        else : p = p_init
        for ii in range(int(X.shape[0]/batch_size)):
            epoch_x = X[ii*batch_size:ii*batch_size+batch_size,:]
            data = epoch_x

            if last_RBM:  pos_hid = np.dot(data,w) + h
            else: pos_hid = 1.0 / (1 + np.exp(np.dot(-data,w) - h))
            batchposhidprobs[ii*batch_size:ii*batch_size+batch_size,:] = pos_hid
            pos_prod = np.dot(data.T, pos_hid)

            pos_hid_act = np.sum(pos_hid,axis = 0)
            pos_vis_act = np.sum(data, axis = 0 ) 
        
            
            if last_RBM:
                pos_hid_binary = (pos_hid + np.random.random((pos_hid.shape))).astype(int)
                neg_data = 1.0/ (1 + np.exp(np.dot(-pos_hid_binary,w.T) - v))
                neg_hid =  np.dot(neg_data,w) + h
            else: 
                pos_hid_binary = (pos_hid > np.random.random((pos_hid.shape))).astype(int)
                neg_data = 1.0/ (1 + np.exp(np.dot(-pos_hid_binary,w.T) - v))
                neg_hid =  1.0/ (1 + np.exp(np.dot(-neg_data,w) - h))
                
            neg_prod = np.dot(neg_data.T,neg_hid)
            neg_hid_act = np.sum(neg_hid, axis = 0)
            neg_vis_act = np.sum(neg_data, axis = 0)

            err = np.sum(np.sum(data - neg_data)**2)
            err_sum += err

            new_w = p * new_w + (e_w * ( (pos_prod - neg_prod) / float(n_classes) - w_decay * w))
            new_v = p * new_v + (e_v * (pos_vis_act - neg_vis_act) / float(n_classes))
            new_h = p * new_h + (e_h * (pos_hid_act - neg_hid_act) / float(n_classes))

            w = w + new_w
            v = v + new_v
            h = h + new_h
        if e % 5 == 0 or e == epochs - 1: print "epoch:%d,    \terror:%1.2e" %(e,err_sum)
    print "shapes:, w, v, h", w.shape, v.shape, h.shape
    print "batchposhidprobs: min,max", np.min(batchposhidprobs), np.max(batchposhidprobs)
    return w,v,h, batchposhidprobs


