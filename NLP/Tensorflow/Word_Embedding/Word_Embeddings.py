import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def cosine_similarity(u, v):
    if np.all(u == v):
        return 1
    
    dot = np.inner(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if(np.isclose(norm_u * norm_v, 0, atol=1e-32)):
        return 0
    
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    word_a = word_a.lower()
    word_b = word_b.lower()
    word_c = word_c.lower()

    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None

    for w in words:
        if w == word_c:
            continue
        cosine_sim = cosine_similarity(e_a - e_b, e_c - word_to_vec_map[w])
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word

def neutralize(word, g, word_to_vec_map):
    e = word_to_vec_map(word)
    e_bias = (np.inner(e, g) * g) / np.linalg.norm(g) ** 2
    e_debiased = e - e_bias

    return e_debiased

def equalize(pair, bias_axis, word_to_vec_map):
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    mu = (e_w1 + e_w2) / 2
    mu_bias = np.inner(mu, bias_axis) * bias_axis / np.linalg.norm(bias_axis) ** 2
    mu_debiased = mu - mu_bias

    e_w1_bias = np.inner(e_w1_bias, bias_axis) * bias_axis / np.linalg.norm(bias_axis) ** 2
    e_w2_bias = np.inner(e_w2_bias, bias_axis) * bias_axis / np.linalg.norm(bias_axis) ** 2

    e_w1_bias_corrected = np.sqrt(1 - np.linalg.norm(mu_debiased) ** 2) * (e_w1_bias - mu_bias) / np.linalg.norm(e_w1_bias - mu_bias)
    e_w2_bias_corrected = np.sqrt(1 - np.linalg.norm(mu_debiased) ** 2) * (e_w2_bias - mu_bias) / np.linalg.norm(e_w2_bias - mu_bias)

    e_w1_debiased = e_w1_bias_corrected + mu_debiased
    e_w2_debiased = e_w2_bias_corrected + mu_debiased

    return e_w1_debiased, e_w2_debiased