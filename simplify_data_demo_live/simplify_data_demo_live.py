"""
Dimensionality reduction reasons:
    1. Space efficiency
    2. Computing efficiency
    3. Visualizations

We'll compare 2 - PCA and T-SNE

feature selection: you select a subset of the original feature set.
feature extraction: you build a new set of features from the original feature set.
"""
import matplotlib.pyplot as pyplot
import numpy

numpy.random.seed(1)
mu_vec1 = numpy.array([0,0,0])
cov_mat1 = numpy.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = numpy.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
print(class1_sample)
mu_vec2 = numpy.array([1,1,1])
cov_mat2 = numpy.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = numpy.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
print(class2_sample)
fig = pyplot.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
pyplot.rcParams['legend.fontsize'] = 10
ax.plot(
    class1_sample[0,:], class1_sample[1,:], class1_sample[2,:],
    'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(
    class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
    '^', markersize=8, alpha=0.5, color='green', label='class2')
ax.legend(loc='upper right')
pyplot.show()

# step 1. take the whole data set ignoring classes make it one big dataset
all_samples = numpy.concatenate((class1_sample, class2_sample), axis=1)
all_samples
all_samples.T

# step 2. compute the N dimensional mean vector, to help compute covariance matrix
mean_x = numpy.mean(all_samples[0,:])
mean_y = numpy.mean(all_samples[1,:])
mean_z = numpy.mean(all_samples[2,:])
mean_vector = numpy.array([[mean_x], [mean_y], [mean_z]])
print('Mean Vector:{}'.format(mean_vector))

# step 3. compute the covariance matrix
cov_mat = numpy.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
print('Covariance Matrix:{}'.format(cov_mat))

# Step 4. compute eigenvectors and eigenvalues
eig_val_sc, eig_vec_sc = numpy.linalg.eig(cov_mat)
for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1, 3).T
    print('Eigenvector {}:{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix:{}'.format(i+1, eig_val_sc[i]))

# step 5. sort eigenvector by decreasing value
eig_pairs = [(
    numpy.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
eig_pairs.sort()
eig_pairs.reverse()
for i in eig_pairs:
    print(i[0])
# choose k eigenvectors w largest eigenvalues to form d x k matrix
matrix_w = numpy.hstack((
    eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
print('Matrix W:\n', matrix_w)

# step 6. use d x k to transform samples to new subspace
transformed = matrix_w.T.dot(all_samples)
assert transformed.shape == (2, 40), 'The matrix is not 2x40 dimensional.'
pyplot.plot(
    transformed[0,0:20], transformed[1,0:20],
    'o', markersize=7, color='green', alpha=0.5, label='class1')
pyplot.plot(
    transformed[0,20:40], transformed[1,20:40],
    '^', markersize=7, color='red', alpha=0.5, label='class2')
pyplot.xlim([-5,5])
pyplot.ylim([-5,5])
pyplot.xlabel('x_values')
pyplot.ylabel('y_values')
pyplot.legend()
pyplot.title('Transformed samples with class labels')
pyplot.show()
