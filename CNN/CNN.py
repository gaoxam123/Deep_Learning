import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + np.squeeze(b)

    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    (f, f, n_c_prev, n_c) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_h = int((n_h_prev + 2 * pad - f) / stride) + 1
    n_w = int((n_w_prev + 2 * pad - f) / stride) + 1

    Z = np.zeros((m, n_h, n_w, n_c))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]

        for h in range(n_h):
            vert_start = stride * h
            vert_end = vert_start + f

            for w in range(n_w):
                horiz_start = stride * w
                horiz_end = horiz_start + f

                for c in range(n_c):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[:, :, :, c]
                    bias = b[:, :, :, c]

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, bias)
    
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

def pool_forward(A_prev, hparameters, mode="max"):
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_h = int((n_h_prev - f) / stride) + 1
    n_w = int((n_w_prev - f) / stride) + 1
    n_c = n_c_prev

    A = np.zeros((m, n_h, n_w, n_c))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            vert_start = stride * h
            vert_end = vert_start + f

            for w in range(n_w):
                horiz_start = stride * w
                horiz_end = horiz_start + f

                for c in range(n_c):
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)

    return A, cache

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    (f, f, n_c_prev, n_c) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    (m, n_h, n_w, n_c) = dZ.shape

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_h):
            vert_start = stride * h
            vert_end = vert_start + f

            for w in range(n_w):
                horiz_start = stride * w
                horiz_end = horiz_start + f

                for c in range(n_c):

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db

def create_mask_from_window(x):
    mask = (x == np.max(x))

    return mask

def distribute_value(dz, shape):
    n_h, n_w = shape
    average = np.prod(shape)

    a = (dz / average) * np.ones(shape)

    return a

def pool_backward(dA, cache, mode="max"):
    A_prev, hparameters = cache

    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_h_prev, n_w_prev, n_c_prev = A_prev.shape
    m, n_h, n_w, n_c = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            vert_start = stride * h
            vert_end = vert_start + f

            for w in range(n_w):
                horiz_start = stride * w
                horiz_end = horiz_start + f
                
                for c in range(n_c):
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += dA[i, h, w, c] * mask

                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    return dA_prev

