import sklearn.preprocessing as pp
import scipy.sparse as sp


mat = sp.rand(100, 1000, 0.8, format='csc')
print(mat)


def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat

print(30*'-')

print(cosine_similarities(mat))

a = np.random.rand(100000, 100)
b = np.random.rand(100)
a_norm = np.linalg.norm(a, axis=1)
b_norm = np.linalg.norm(b)
(a @ b) / (a_norm * b_norm)

p set(doc_topic_sims_old.keys())-set(doc_topic_sims.keys())
p set(doc_topic_sims.values())-set(doc_topic_sims_old.values())