#%%
#Libraries
import numpy as np
import h5py 
import matplotlib.pyplot as plt
from math import sqrt, pi
from sklearn.cluster import KMeans

#%%
#Read
f = h5py.File('../data/assignmentSegmentBrainGmmEmMrf.mat','r') 
list(f.keys())
imageData = f.get('imageData')
imageMask = f.get('imageMask')
imageData = np.array(imageData)
imageData = imageData.T
imageMask = np.array(imageMask)
imageMask = imageMask.T

#%%
#GMM With EM
def label_priors(input_label,old_labels,beta,map_left,
    map_right,map_top,map_bottom,mask):
    img = input_label
    if len(input_label)==1:
        img = input_label*np.ones((old_labels.shape))
    top = ((img-np.roll(old_labels,[1,1], [0,1]))*map_top)!=0
    bottom = ((img-np.roll(old_labels,[-1,1], [0,1]))*map_bottom)!=0
    left = ((img-np.roll(old_labels,[1,2], [0,1]))*map_left)!=0
    right = ((img-np.roll(old_labels,[-1,2], [0,1]))*map_right)!=0
    return np.exp(-((top+bottom+left+right)*beta))*mask

def memberships( y,means,sigmas,old_labels,mask,prior_val):
    K = means.shape[0]
    likelihood = np.zeros((y.shape[0], y.shape[1], 3))
    prior = np.zeros((y.shape[0], y.shape[1], 3))
    for i in range(K):
        likelihood[:,:,i] = ((1/(sigmas[i,0]*sqrt(2*pi)))*np.exp(-(y-means[i,0])**2/(2*sigmas[i,0]**2)))*mask
        prior[:,:,i] = prior_val(np.array([i]),old_labels)    
    norm = np.sum(prior,2)
    for i in range(K):
        prior[:,:,i] /=norm 
    membership = likelihood*prior
    norm = np.sum(membership,2)
    for i in range(K):
        membership[:,:,i] /= norm
    
    temp = np.zeros((old_labels.shape))
    for i in range(K):
        temp = membership[:,:,i]
        temp[mask==0] = 0
        membership[:,:,i] = temp 
        
    return membership
    
def gaussian_paprameters(y, mem, maps ):
    means = np.zeros((mem.shape[2],1))
    sigmas = np.zeros((mem.shape[2],1)) 
    for i in range(mem.shape[2]):
        den = np.sum(mem[:, :, i]) 
        means[i,0] = np.sum(mem[:, :, i]*y)/den
        sigmas[i,0] = np.sqrt(np.sum(mem[:,:,i]*((y-means[i,0])**2)*maps)/den)
        
    return means, sigmas    
    
def posterior_val(old_labels, y, means,sigmas, maps, prior_val):
    likelihood = np.zeros((old_labels.shape))
    for i in range(len(means)):
        indeold_labels = np.where(old_labels==i)
        likelihood[indeold_labels] = (1/(sigmas[i,0]*sqrt(2*pi)))*np.exp(-(y[indeold_labels]-means[i,0])**2/(2*(sigmas[i,0]**2)))
    prior = prior_val(old_labels,old_labels)
    return likelihood*prior*maps   
   
def segmentation( old_labels,y,means,sigmas,iters,mask,prior_val):
    for i in range(iters):
        oldLogPosterior = np.sum(np.log(posterior_val(old_labels,y,means,sigmas,mask,prior_val)[mask!=0]))
        print('%d : Initial log posterior = %f\n'%(i,oldLogPosterior))
        membership = memberships(y,means,sigmas,old_labels,mask,prior_val)
        new_labels = np.argmax(membership,2)
        new_labels = new_labels*mask
        posterior = posterior_val(new_labels,y,means,sigmas,mask,prior_val)
        newLogPosterior = np.sum(np.log(posterior[mask!=0]))
        print('%d : Final log posterior = %f\n'%(i,newLogPosterior))
        
        if newLogPosterior<oldLogPosterior:
            break
        means,sigmas = gaussian_paprameters(y,membership,mask)
        equal = np.array_equal(old_labels, new_labels)
        if equal:
            break
        old_labels = new_labels
        
    return old_labels,means,sigmas,iters
    

#%%
#Label initialization
s = imageData.shape
K = 3
map_left = np.roll(imageMask, [1,2], [0,1])
map_right = np.roll(imageMask, [-1,2], [0,1])
map_top = np.roll(imageMask, [1,1], [0,1])
map_bottom = np.roll(imageMask, [-1,1], [0,1])

beta1 = 2
beta2 = 0  

prior1 = lambda input_label,old_labels: label_priors(
    input_label,old_labels,beta1,map_left,map_right,
    map_top,map_bottom,imageMask)

prior2 = lambda input_label,old_labels: label_priors(
    input_label,old_labels,beta2,map_left,map_right,
    map_top,map_bottom,imageMask)


masked_img = imageData[imageMask>0]
masked_img = np.reshape(masked_img, (masked_img.shape[0], 1))
kmeans = KMeans(n_clusters = K).fit(masked_img)
initial_labels = kmeans.labels_
means_init = kmeans.cluster_centers_
label_map = np.zeros((imageData.shape))
label_map[imageMask>0] = initial_labels


#%%
#Variance
sigmas_init = np.zeros((K,1))
for i in range(K):
    clusterVals = masked_img[initial_labels==i]
    sigmas_init[i] = np.linalg.norm(clusterVals - means_init[i])/sqrt(len(clusterVals))

#%% 
#Segmentation
initial_labels = label_map
print('Modified ICM with beta = %f \n'%(beta1))
labels1,means1,sigmas1,iters1 = segmentation(initial_labels,imageData,means_init,
    sigmas_init,20,imageMask,prior1)

print('Modified ICM with beta = %f \n'%(beta2))
labels2,means2,sigmas2,iters2 = segmentation(initial_labels,imageData,means_init,
    sigmas_init,20,imageMask,prior2)

#%%
#Images
plt.figure()
plt.imshow(imageData, cmap=plt.cm.gray)
plt.title('Corrupted image')

plt.figure()
plt.imshow(initial_labels, extent=[0, 1, 0, 1])
plt.title('Initial estimate for the label image')


plt.figure()
plt.imshow(labels1, extent=[0, 1, 0, 1])
plt.title('Optimal label image estimate for beta = 2.0')

plt.figure()
plt.imshow(labels2, extent=[0, 1, 0, 1])
plt.title('Optimal label image estimate for beta = 0')

#Beta = 2.0
for i in range(K):
    seg = np.zeros((imageData.shape))
    seg[labels1==i] = imageData[labels1==i]
    plt.figure()
    plt.imshow(seg, cmap=plt.cm.gray)
    plt.title('Optimal class membership image estimate %d for beta = 2.0'%(i+1))
    
#Beta =0
for i in range(K):
    seg = np.zeros((imageData.shape))
    seg[labels2==i] = imageData[labels2==i]
    plt.figure()
    plt.imshow(seg, cmap=plt.cm.gray)
    plt.title('Optimal class membership image estimate %d for beta = 0'%(i+1))
 
#%%
#Optimal Estimates    
print('Initial class means are (%f, %f, %f)'%(means_init[0,0],means_init[1,0],means_init[2,0]))
print('For beta = 2.0, optimal class means are (%f, %f, %f)'%(means1[0,0],means1[1,0],means1[2,0]))
print('For beta = 0, optimal class means are (%f, %f, %f)'%(means2[0,0],means2[1,0],means2[2,0]))



