import numpy as np 

def get_limbs_ntu(motion):
    J, D, T = motion.shape
    limbs = np.zeros([24, D, T])

    limbs[0] = motion[20]-motion[1]
    limbs[1] = motion[2] -motion[20]
    limbs[2] = motion[3] -motion[2]
    limbs[3] = motion[8] -motion[20]
    limbs[4] = motion[9] -motion[8]
    limbs[5] = motion[10]-motion[9]
    limbs[6] = motion[24]-motion[10]
    limbs[7] = motion[11]-motion[10]
    limbs[8] = motion[23]-motion[11]

    limbs[9] = motion[4]-motion[20]
    limbs[10]= motion[5]-motion[4]
    limbs[11]= motion[6]-motion[5]
    limbs[12]= motion[22]-motion[6]
    limbs[13]= motion[7]-motion[6]
    limbs[14]= motion[21]-motion[7]

    limbs[15]= motion[0]-motion[1]
    limbs[16]= motion[16]-motion[0]
    limbs[17]= motion[17]-motion[16]
    limbs[18]= motion[18]-motion[17]
    limbs[19]= motion[19]-motion[18]

    limbs[20]= motion[12]-motion[0]
    limbs[21]= motion[13]-motion[12]
    limbs[22]= motion[14]-motion[13]
    limbs[23]= motion[15]-motion[14]

    return limbs



def random_scale_limb(data_numpy, global_scale, local_scales):
    '''
    
    '''
    C,T,V,M=data_numpy.shape
    assert local_scales.shape[0]==V-1
    data_numpy=np.transpose(data_numpy,[2,0,1,3]) # (V,C,T,M)
    data_numpy=data_numpy.reshape((V,C,T*M)) # 
    res=scale_limbs_ntu(data_numpy, global_scale, local_scales) 
    res=res.reshape((V,C,T,M))
    res=np.transpose(res,[1,2,0,3])
    # print(data_numpy.shape, res.shape)
    return res


def scale_limbs_ntu(motion, global_scale, local_scales):
    """
    :param motion: joint sequence [J, 2, T]
    :param local_scales: 8 numbers of scales
    :return: scaled joint sequence
    """

    limb_dependents = {
       
       0: [20,2,3,8,9,10,11,24,23, 4,5,6,7,21,22],
       1: [2,3],
       2: [3],
       3: [8,9,10,11,23,24],
       4: [9,10,11,23,24],
       5: [10,11,23,24],
       6: [24],
       7: [11,23],
       8: [23],
       9: [4,5,6,7,21,22],
       10: [5,6,7,21,22],
       11: [6,7,21,22],
       12: [22],
       13: [7,21],
       14: [21],
       15: [0,12,13,14,15,16,17,18,19],
       16: [16,17,18,19],
       17: [17,18,19],
       18: [18,19],
       19: [19],
       20: [12,13,14,15],
       21: [13,14,15],
       22: [14,15],
       23: [15]
    }

    limbs = get_limbs_ntu(motion)

    
    scaled_limbs = limbs.copy() * global_scale
    n=len(local_scales)
    for i in range(n):
        scaled_limbs[i] *= local_scales[i]

    # embed()
    
    delta = scaled_limbs - limbs

    # embed()

    scaled_motion = motion.copy()

    for i in range(24):
        scaled_motion[limb_dependents[i]] += delta[i]

    return scaled_motion





def get_bone_length(data_numpy):
    '''
    args:   (c,t,v,m)
    return: (  t,n_bones)
    '''
    

    C,T,V,M=data_numpy.shape
    assert M==1
    data_numpy=np.transpose(data_numpy,[2,0,1,3]) # (V,C,T,M)
    data_numpy=data_numpy.reshape((V,C,T*M)) # 

    # np.zeros([24, D, T])
    limbs = get_limbs_ntu(data_numpy)
    limbs_length = np.linalg.norm(limbs, axis=1)

    limbs_length = limbs_length.T
    return limbs_length


def batch_get_bone_length(data_batch):
    res=[]
    n,c,t,v,m=data_batch.shape
    for i in range(n):
        data_numpy=data_batch[i]
        limbs_length=get_bone_length(data_numpy)
        res.append(limbs_length)
    res=np.array(res)
    return res 


def batch_random_scale_limb(data_batch, batch_global_scale, batch_local_scales):

    assert len(data_batch)==len(batch_global_scale)
    assert len(data_batch)==len(batch_local_scales)
    
    res=[]
    bs = len(data_batch)
    for i in range(bs):
        data_numpy=data_batch[i]
        global_scale=batch_global_scale[i]
        local_scales=batch_local_scales[i]
        retargeted=random_scale_limb(data_numpy, global_scale, local_scales)
        res.append(retargeted)
    res = np.array(res)
    return res 














# scale on (joint, time) 2 dims.
def random_scale_limb_vt(data_numpy, global_scale, local_scales):
    '''
    local_scales: (24,5)
    '''
    C,T,V,M=data_numpy.shape
    assert local_scales.shape[0]==V-1
    assert len(local_scales.shape)==2 and local_scales.shape[1]==T

    data_numpy=np.transpose(data_numpy,[2,0,1,3]) # (V,C,T,M)
    data_numpy=data_numpy.reshape((V,C,T*M)) # 
    res=scale_limbs_ntu_vt(data_numpy, global_scale, local_scales) 
    res=res.reshape((V,C,T,M))
    res=np.transpose(res,[1,2,0,3])
    # print(data_numpy.shape, res.shape)
    return res


def scale_limbs_ntu_vt(motion, global_scale, local_scales):
    """
    :param motion: joint sequence [J, 2, T]
    :param local_scales: 8 numbers of scales
    :return: scaled joint sequence
    """

    limb_dependents = {
       
       0: [20,2,3,8,9,10,11,24,23, 4,5,6,7,21,22],
       1: [2,3],
       2: [3],
       3: [8,9,10,11,23,24],
       4: [9,10,11,23,24],
       5: [10,11,23,24],
       6: [24],
       7: [11,23],
       8: [23],
       9: [4,5,6,7,21,22],
       10: [5,6,7,21,22],
       11: [6,7,21,22],
       12: [22],
       13: [7,21],
       14: [21],
       15: [0,12,13,14,15,16,17,18,19],
       16: [16,17,18,19],
       17: [17,18,19],
       18: [18,19],
       19: [19],
       20: [12,13,14,15],
       21: [13,14,15],
       22: [14,15],
       23: [15]
    }

    limbs = get_limbs_ntu(motion)

    local_scales = local_scales[:,None,:]

    # (v,c,t)
    scaled_limbs = limbs.copy() * global_scale
    n=len(local_scales)
    for i in range(n):
        scaled_limbs[i] *= local_scales[i]

    # embed()
    
    delta = scaled_limbs - limbs

    # embed()

    scaled_motion = motion.copy()

    for i in range(24):
        scaled_motion[limb_dependents[i]] += delta[i]

    return scaled_motion

def batch_random_scale_limb_vt(data_batch, batch_global_scale, batch_local_scales):

    assert len(data_batch)==len(batch_global_scale)
    assert len(data_batch)==len(batch_local_scales)
    
    res=[]
    bs = len(data_batch)
    for i in range(bs):
        data_numpy=data_batch[i]
        global_scale=batch_global_scale[i]
        local_scales=batch_local_scales[i]
        retargeted=random_scale_limb_vt(data_numpy, global_scale, local_scales)
        res.append(retargeted)
    res = np.array(res)
    return res 
