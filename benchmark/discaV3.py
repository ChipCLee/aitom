from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn import init
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA    
import sys, multiprocessing, importlib, pickle, time
from multiprocessing.pool import Pool  
from tqdm.auto import tqdm
from torchvision.transforms import Normalize
import os

from aitom.classify.deep.unsupervised.disca.util import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import homogeneity_completeness_v_measure


import warnings
warnings.filterwarnings("ignore")
image_size = 24
# configuration
class Config:
    data_dir = ''
    image_size = 24 ### subtomogram size ###
    candidateKs = [2,3,4,5,6,7,8]  ### candidate number of clusters to test, it is also possible to set just one large K that overpartites the data

    batch_size = 64
    # M = 20  ### number of iterations ###
    M = 20  ### number of iterations ###
    lr = 1e-5  ### testing CNN learning rate ###
    # lr = 1e-5  ### Original CNN learning rate ###

    label_smoothing_factor = 0.2  ### label smoothing factor ###
    reg_covar = 0.00001

    model_path = '/home/mohamad.kassab/aitom/model_torch.pth'  ### path for saving torch model, should be a pth file ###
    label_path = '/home/mohamad.kassab/aitom/label_path_torch.pickle'  ### path for saving labels, should be a .pickle file ###

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Subtomogram_Dataset:
    def __init__(self, train_data, label_one_hot):
        self.train_data = train_data
        self.label_one_hot = label_one_hot

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        features = self.train_data[index]
        labels = self.label_one_hot[index]

        features = torch.FloatTensor(features)
        labels = torch.FloatTensor(labels)

        return features, labels

def align_cluster_index(ref_cluster, map_cluster):
    """                                                                                                                                                                            
    remap cluster index according the the ref_cluster.                                                                                                                                    
    both inputs must have same number of unique cluster index values.                                                                                                                      
    """

    ref_values = np.unique(ref_cluster)
    map_values = np.unique(map_cluster)

    if ref_values.shape[0] != map_values.shape[0]:
        print('error: both inputs must have same number of unique cluster index values.')
        return ()
    cont_mat = contingency_matrix(ref_cluster, map_cluster)

    row_ind, col_ind = linear_sum_assignment(len(ref_cluster) - cont_mat)

    map_cluster_out = map_cluster.copy()

    for i in ref_values:
        map_cluster_out[map_cluster == col_ind[i]] = i

    return map_cluster_out



def DDBI(features, labels):
    """                                                                                                                                                                            
    compute the Distortion-based Davies-Bouldin index defined in Equ 1 of the Supporting Information.                                                                                                        
    """

    means_init = np.array([np.mean(features[labels == i], 0) for i in np.unique(labels)])
    precisions_init = np.array(
        [np.linalg.inv(np.cov(features[labels == i].T) + Config.reg_covar * np.eye(features.shape[1])) for i in
         np.unique(labels)])

    T = np.array([np.mean(np.diag(
        (features[labels == i] - means_init[i]).dot(precisions_init[i]).dot((features[labels == i] - means_init[i]).T)))
                  for i in np.unique(labels)])

    D = np.array(
        [np.diag((means_init - means_init[i]).dot(precisions_init[i]).dot((means_init - means_init[i]).T)) for i in
         np.unique(labels)])

    DBI_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))

    for i in range(len(np.unique(labels))):
        for j in range(len(np.unique(labels))):
            if i != j:
                DBI_matrix[i, j] = (T[i] + T[j]) / (D[i, j] + D[j, i])

    DBI = np.mean(np.max(DBI_matrix, 0))

    return DBI

class YOPOFeatureModel(nn.Module):

    def __init__(self):
        super(YOPOFeatureModel, self).__init__()

        self.dropout = nn.Dropout(0.5)
        self.m1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m2 = nn.Conv3d(64, 80, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m3 = nn.Conv3d(80, 96, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m4 = nn.Conv3d(96, 112, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m5 = nn.Conv3d(112, 128, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m6 = nn.Conv3d(128, 144, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m7 = nn.Conv3d(144, 160, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m8 = nn.Conv3d(160, 176, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m9 = nn.Conv3d(176, 192, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.m10 = nn.Conv3d(192, 208, kernel_size=(3, 3, 3), dilation=(1, 1, 1), padding=0)
        self.elu = nn.ELU()
        self.global_max_pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.batch_norm1 = nn.BatchNorm3d(64)
        self.batch_norm2 = nn.BatchNorm3d(80)
        self.batch_norm3 = nn.BatchNorm3d(96)
        self.batch_norm4 = nn.BatchNorm3d(112)
        self.batch_norm5 = nn.BatchNorm3d(128)
        self.batch_norm6 = nn.BatchNorm3d(144)
        self.batch_norm7 = nn.BatchNorm3d(160)
        self.batch_norm8 = nn.BatchNorm3d(176)
        self.batch_norm9 = nn.BatchNorm3d(192)
        self.batch_norm10 = nn.BatchNorm3d(208)




        # self.m11 = self.get_block(104, 117)
        # self.m12 = self.get_block(117, 140)
        # self.m13 = self.get_block(140, 150)
        self.batchnorm = torch.nn.BatchNorm3d(1360)
        self.linear = nn.Linear(
            in_features=1360,
            out_features=1024
        )
       
        self.weight_init(self)
        
    '''
	Initialising the model with blocks of layers.
	'''

    @staticmethod
    def get_block(input_channel_size, output_channel_size):
        return nn.Sequential(
            torch.nn.Conv3d(in_channels=input_channel_size,
                            out_channels=output_channel_size,
                            kernel_size=(3, 3, 3),
                            padding=0,
                            dilation=(1, 1, 1)),  
            #torch.nn.BatchNorm3d(output_channel_size),
            #torch.nn.ELU(inplace=True),
            # torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    '''
	Initialising weights of the model.
	'''

    @staticmethod
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    '''
	Forward Propagation Pass.
	'''

    def forward(self, input_image):
        output  = input_image.view(-1, 1, image_size, image_size, image_size)        
        output = self.dropout(output)

        output = self.m1(output)
        output = self.elu(output)
        output = self.batch_norm1(output)
        o1 = self.global_max_pooling(output)


        output = self.m2(output)
        output = self.elu(output)
        output = self.batch_norm2(output)
        o2 = self.global_max_pooling(output)

        output = self.m3(output)
        output = self.elu(output)
        output = self.batch_norm3(output)
        o3 = self.global_max_pooling(output)

        output = self.m4(output)
        output = self.elu(output)
        output = self.batch_norm4(output)
        o4 = self.global_max_pooling(output)


        output = self.m5(output)
        output = self.elu(output)
        output = self.batch_norm5(output)
        o5 = self.global_max_pooling(output)

        output = self.m6(output)
        output = self.elu(output)
        output = self.batch_norm6(output)
        o6 = self.global_max_pooling(output)


        output = self.m7(output)
        output = self.elu(output)
        output = self.batch_norm7(output)
        o7 = self.global_max_pooling(output)

        output = self.m8(output)
        output = self.elu(output)
        output = self.batch_norm8(output)
        o8 = self.global_max_pooling(output)

        output = self.m9(output)
        output = self.elu(output)
        output = self.batch_norm9(output)
        o9 = self.global_max_pooling(output)


        output = self.m10(output)
        output = self.elu(output)
        output = self.batch_norm10(output)
        o10 = self.global_max_pooling(output)












          # print(output.size())
        """
		output = self.m11(output)
		o11 = F.max_pool3d(output, kernel_size=output.size()[2:])
		output = self.m12(output)
		o12 = F.max_pool3d(output, kernel_size=output.size()[2:])
		output = self.m13(output)
		o13 = F.max_pool3d(output, kernel_size=output.size()[2:])
		"""
        m = torch.cat((o1, o2, o3, o4, o5, o6, o7, o8, o9, o10), dim=1)
        # print(m.size())
        m = self.batchnorm(m)
        m = nn.Flatten()(m)
        m = self.linear(m)
        return m


def statistical_fitting(features, labels, candidateKs, K, reg_covar, i):
    """
    fitting a Gaussian mixture model to the extracted features from YOPO
    given current estimated labels, K, and a number of candidateKs. 

    reg_covar: non-negative regularization added to the diagonal of covariance.     
    i: random_state for initializing the parameters.
    """

    pca = PCA(n_components=16)  
    features_pca = pca.fit_transform(features) 

    labels_K = [] 
    BICs = [] 
                                                                                                                                                            
    for k in candidateKs: 
        if k == K: 
            try:
                weights_init = np.array([np.sum(labels == j)/float(len(labels)) for j in range(k)]) 
                means_init = np.array([np.mean(features_pca[labels == j], 0) for j in range(k)]) 
                precisions_init = np.array([np.linalg.inv(np.cov(features_pca[labels == j].T) + reg_covar * np.eye(features_pca.shape[1])) for j in range(k)]) 
 
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i,  
                                        weights_init=weights_init, means_init=means_init, precisions_init=precisions_init, init_params = 'random') 
 
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca)

            except:     
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca) 
                         
         
            gmm_1 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
            gmm_1.fit(features_pca) 
            labels_k_1 = gmm_1.predict(features_pca) 
             
            m_select = np.argmin([gmm_0.bic(features_pca), gmm_1.bic(features_pca)]) 
             
            if m_select == 0: 
                labels_K.append(labels_k_0) 
                 
                BICs.append(gmm_0.bic(features_pca)) 
             
            else: 
                labels_K.append(labels_k_1) 
                 
                BICs.append(gmm_1.bic(features_pca)) 
         
        else: 
            gmm = GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, reg_covar=reg_covar, max_iter=100, n_init=5, random_state=i, init_params = 'random') 
         
            gmm.fit(features_pca) 
            labels_k = gmm.predict(features_pca) 

            labels_K.append(labels_k) 
             
            BICs.append(gmm.bic(features_pca)) 
    
    labels_temp = remove_empty_cluster(labels_K[np.argmin(BICs)])                     
     
    K_temp = len(np.unique(labels_temp)) 
     
    if K_temp == K: 
        same_K = True 
    else: 
        same_K = False 
        K = K_temp     

    print('Estimated K:', K)
    
    return labels_temp, K, same_K, features_pca   



def convergence_check(i, M, labels_temp, labels, done):
    if i > 0:
        if np.sum(labels_temp == labels) / float(len(labels)) > 1:
        # if np.sum(labels_temp == labels) / float(len(labels)) > 0.999:
            done = True

    i += 1
    if i == M:
        done = True

    labels = labels_temp

    return i, labels, done



def pickle_dump(o, path, protocol=2):
    """                                                                                                                                                                            
    write a pickle file given the object o and the path.                                                                                                                      
    """ 
    with open(path, 'wb') as f:    pickle.dump(o, f, protocol=protocol)



def run_iterator(tasks, worker_num=multiprocessing.cpu_count(), verbose=False):
    """
    parallel multiprocessing for a given task, this is useful for speeding up the data augmentation step.
    """

    if verbose:		print('parallel_multiprocessing()', 'start', time.time())

    worker_num = min(worker_num, multiprocessing.cpu_count())

    for i,t in tasks.items():
        if 'args' not in t:     t['args'] = ()
        if 'kwargs' not in t:     t['kwargs'] = {}
        if 'id' not in t:   t['id'] = i
        assert t['id'] == i

    completed_count = 0 
    if worker_num > 1:

        pool = Pool(processes = worker_num)
        pool_apply = []
        for i,t in tasks.items():
            aa = pool.apply_async(func=call_func, kwds={'t':t})

            pool_apply.append(aa)


        for pa in pool_apply:
            yield pa.get(99999)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()

        pool.close()
        pool.join()
        del pool

    else:

        for i,t in tasks.items():
            yield call_func(t)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()
	
    if verbose:		print('parallel_multiprocessing()', 'end', time.time())


    
run_batch = run_iterator #alias



def call_func(t):

    if 'func' in t:
        assert 'module' not in t
        assert 'method' not in t
        func = t['func']
    else:
        modu = importlib.import_module(t['module'])
        func = getattr(modu, t['method'])

    r = func(*t['args'], **t['kwargs'])
    return {'id':t['id'], 'result':r}



def random_rotation_matrix():
    """
    generate a random 3D rigid rotation matrix.
    """
    m = np.random.random( (3,3) )
    u,s,v = np.linalg.svd(m)

    return u



def rotate3d_zyz(data, Inv_R, center=None, order=2):
    """
    rotate a 3D data using ZYZ convention (phi: z1, the: x, psi: z2).
    """
    # Figure out the rotation center
    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        assert len(center) == 3
        (cx, cy, cz) = center

    
    from scipy import mgrid
    grid = mgrid[-cx:data.shape[0]-cx, -cy:data.shape[1]-cy, -cz:data.shape[2]-cz]
    # temp = grid.reshape((3, np.int(grid.size / 3)))
    temp = grid.reshape((3, int(grid.size / 3)))
    temp = np.dot(Inv_R, temp)
    grid = np.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz

    # Interpolation
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order)

    return d



def data_augmentation(x_train, factor = 2):
    """
    data augmentation given the training subtomogram data.
    if factor = 1, this function will return the unaugmented subtomogram data.
    if factor > 1, this function will return (factor - 1) number of copies of augmented subtomogram data.
    """

    if factor > 1:

        x_train_augmented = []
        
        x_train_augmented.append(x_train)

        for f in range(1, factor):
            ts = {}        
            for i in range(len(x_train)):                       
                t = {}                                                
                t['func'] = rotate3d_zyz                                   
                                                      
                # prepare keyword arguments                                                                                                               
                args_t = {}                                                                                                                               
                # args_t['data'] = x_train[i,:,:,:,0]   
                args_t['data'] = x_train[i,0,:,:,:]                                                                                                                 
                args_t['Inv_R'] = random_rotation_matrix()                                                   
                                                                                                                                                                                                                                           
                t['kwargs'] = args_t                                                  
                ts[i] = t                                                       
                                                                      
            rs = run_batch(ts, worker_num=48)
            # x_train_f = np.expand_dims(np.array([_['result'] for _ in rs]), -1)
            x_train_f = np.expand_dims(np.array([_['result'] for _ in rs]), 1)
            
            # print(x_train_f.shape)
            x_train_augmented.append(x_train_f)
            
        x_train_augmented = np.concatenate(x_train_augmented)
    
    else:
        x_train_augmented = x_train                        

        x_train[x_train == 0] = np.random.normal(loc=0.0, scale=1.0, size = np.sum(x_train == 0))

    return x_train_augmented



def one_hot(a, num_classes):
    """
    one-hot encoding. 
    """
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])   



def smooth_labels(labels, factor=0.1):
    """
    label smoothing. 
    """
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
 
    return labels           



def remove_empty_cluster(labels):
    """
    if there are no samples in a cluster,
    this function will remove the cluster and make the remaining cluster number compact. 
    """
    labels_unique = np.unique(labels)
    for i in range(len(np.unique(labels))):
        labels[labels == labels_unique[i]] = i

    return labels



def prepare_training_data(x_train, labels, label_smoothing_factor):
    """
    training data preparation given the current training data x_train, labels, and label_smoothing_factor
    """

    label_one_hot = one_hot(labels, len(np.unique(labels))) 
    
    factor_use = 1
    index = np.array(range(x_train.shape[0] * factor_use)) 

    np.random.shuffle(index)         
     
    x_train_augmented = data_augmentation(x_train, factor_use) 
    
    x_train_permute = x_train_augmented[index].copy() 

    label_smoothing_factor *= 0.9 

    labels_augmented = np.tile(smooth_labels(label_one_hot, label_smoothing_factor), (2,1))               

    labels_permute = labels_augmented[index].copy() 

    return label_one_hot, x_train_permute, label_smoothing_factor, labels_permute



class YOPOClassification(nn.Module):
    def __init__(self, num_labels, vector_size=1024):
        super(YOPOClassification, self).__init__()
        self.main_input = nn.Linear(vector_size, num_labels)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.main_input(x)
        x = self.softmax(x)
        return x

class YOPO_Final_Model(nn.Module):
    def __init__(self, yopo_feature, yopo_classification):
        super(YOPO_Final_Model, self).__init__()
        self.feature_model = yopo_feature
        self.classification_model = yopo_classification
        
    def forward(self, input_image):
        features = self.feature_model(input_image)
        output = self.classification_model(features)
        return output

def update_output_layer(K, label_one_hot, batch_size, model_feature, features, lr, verbose=False):
    print('Updating output layer')
    model_classification = YOPOClassification(num_labels=K).to(Config.device)

    optim = torch.optim.NAdam(model_classification.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.MultiMarginLoss()

    # Convert features and label_one_hot to PyTorch tensors

    dataset = Subtomogram_Dataset(features, label_one_hot)

    # Create a DataLoader for batch processing
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_loss = []
    

    for epoch in range(10):
        model_classification.train()
        train_total = 0.0
        train_correct = 0.0
        epoch_loss = 0.0
        start_time = time.time()
        pbar = tqdm(train_loader, desc = 'Iterating over train data, Epoch: {}/{}'.format(epoch + 1, 10))
        for features, labels in pbar:
            features = features.to(Config.device)
            labels = labels.to(Config.device)

            pred = model_classification(features)
            
            # print(pred.shape)
            # time.sleep(5)

            optim.zero_grad()

            predicted = torch.argmax(pred, 1)
            labels_1 = torch.argmax(labels, 1)

            loss = criterion(pred, labels_1)
            epoch_loss += loss.item()

            loss.backward()
            optim.step()
            
            train_correct += (predicted == labels_1).sum().float().item()
            train_total += labels.size(0)

        exec_time = time.time() - start_time
        model_loss.append(epoch_loss)
        
        # calculate accuracy
        # print(train_correct, train_total)
        accuracy = train_correct / train_total
        
        if verbose:
            print('Epoch: {}/{} Loss: {:.4f} accuracy: {:.4f} In: {:.4f}s'.format(epoch + 1, 10, epoch_loss, accuracy, exec_time))
    
    ### New YOPO ### 
    # model = nn.Sequential(
    #     *list(model_feature.children())[:-1],  # Use all layers of model_feature except the last one
    #     model_classification
    # )

    model = YOPO_Final_Model(model_feature, model_classification)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.MultiMarginLoss()
    
    # print(model)

    print('Output layer updated')
    return model, optimizer, criterion

def image_normalization(img_list):
    ### img_list is a list cantains images, returns a list contains normalized images

    normalized_images = []
    print('Normalizing')
    for image in img_list:
        image = np.array(image)
        image = torch.tensor(image)
        normalize_single = Normalize(mean=[image.mean()], std=[image.std()])(image).tolist()
        normalized_images.append(normalize_single)
    print('Normalizing finished')
    return normalized_images




if __name__ == '__main__':  

   
    path_v='/l/users/mohamad.kassab/simulated/DISCA_DATA_60_0.1_v.pickle'

    with open(path_v, 'rb', 0) as f:
        data = pickle.load(f, encoding= 'latin1')

    #v = [_['v'] for _ in data]
    v = np.array(v)

    x_train = np.expand_dims(v, -1)

    #x_keys = [_ for _ in data_sv['vs'] if data_sv['vs'][_]['v'] is not None]
    #x_train = [np.expand_dims(data_sv['vs'][_]['v'], -1) for _ in x_keys]
    #data_array_normalized = np.array(x_train)

    #print(x_train[0].mean())

    #data_array_normalized = np.array(x_train)
    #data_array_normalized = np.random.shuffle(data_array_normalized)
    data_array_normalized = x_train

    print('x_train.shape:',data_array_normalized.shape)

    # ### Load Dataset Here ###     
    # with open('filename.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # # print(data)
    # data_array_normalized = []
    # for i in range(data.shape[0]):
    #     x = data[i]
    #     x = (x - np.mean(x))/np.std(x)
    #     data_array_normalized.append(x)
    # data_array_normalized = np.array(data_array_normalized).reshape(data.shape[0],1,data.shape[1],data.shape[2], data.shape[3])
    # print(data_array_normalized.shape)
    
    
    x_train = torch.tensor(data_array_normalized, dtype=torch.float32)  ### load the x_train data, should be shape (n, 1, shape_1, shape_2, shape_3)

    path_id='/l/users/mohamad.kassab/final_data/DISCA_DATA_60_0.1_id.pickle'
    with open(path_id, 'rb') as f:
        gt_data = pickle.load(f)

    
    #class_mapping = {'1i6v': 0, '1qo0': 1, '3dy4': 2, '4v4a-pdb-bundle1': 3, '5lqw': 4}

    class_mapping = {'1I6V': 0, '1QO1': 1, '3DY4': 2, '4V4A': 3, '5LQW': 4}

    #class_mapping = {'1BXR':0, '1EQR':1, '1KP8':2, '1QO1':3, '1VPX':4, '2GLS':5, '3DY4':6, '4V4A':7,'5LQW':8, '6A5L':9}
    #numerical_ground_truth = [class_mapping[_] for _ in gt_data]
    
    gt = numerical_ground_truth  ### load or define label ground truth here, if for simulated data
    
    ### Generalized EM Process ###
    K = None
    labels = None
    DBI_best = float('inf')

    done = False
    i = 0

    total_loss = []

    iterations = []
    losses = []
    accuracies = []
    execution_times = []
    learning_rate = []

    # milestones = [65]  # Add the milestones where you want to change the learning rate
    # scheduler = MultiStepLR(optim, milestones=milestones, gamma=0.1)


    while not done:
        print('Iteration:', i)
        
    # feature extraction

        if i == 0:
            model_feature = YOPOFeatureModel().to(Config.device)
        else:
            model_feature = nn.Sequential(*list(model.children())[:-1])


        
        criterion = nn.MultiMarginLoss()
        optim = torch.optim.Adam(model_feature.parameters(), lr=Config.lr)

        features = np.empty((0, 1024))
        train_input_loader = DataLoader(x_train, batch_size = Config.batch_size, shuffle = False)
        
        train_pbar = tqdm(train_input_loader, desc='Feature extraction')
        model_feature.eval()    
        with torch.no_grad():
            for batch in train_pbar:
                batch = batch.to(Config.device)
                temp_features = model_feature(batch).detach().cpu().numpy()
                features = np.append(features, temp_features, axis=0) 

    ### Feature Clustering ###

        labels_temp, K, same_K, features_pca = statistical_fitting(features=features, labels=labels, candidateKs=Config.candidateKs, K=K, reg_covar=Config.reg_covar, i = i)
        
    ### Matching Clusters by Hungarian Algorithm ###

        if same_K:
            labels_temp = align_cluster_index(labels, labels_temp)

        i, labels, done = convergence_check(i=i, M=Config.M, labels_temp=labels_temp, labels=labels, done=done)

        print('Cluster sizes:', [np.sum(labels == k) for k in range(K)])

    ### Validate Clustering by distortion-based DBI ###

        DBI = DDBI(features_pca, labels)

        if DBI < DBI_best:
        # if DBI >= DBI_best:
            #if i > 1:
                #torch.save(model, Config.model_path)  ### save model here ###
                #print(f'Best Model Saved to: {Config.model_path}')

                #pickle_dump(labels, Config.label_path)

            labels_best = labels  ### save current labels if DDBI improves ###

            DBI_best = DBI

        print('DDBI:', DBI, '############################################')

    ## Permute Samples ###   
        print('Prepearing training data')
        label_one_hot, x_train_permute, label_smoothing_factor, labels_permute = prepare_training_data(x_train=data_array_normalized, labels=labels, label_smoothing_factor=Config.label_smoothing_factor)
        print('Finished')
    ### Finetune new model with current estimated K ### 
        if not same_K: 
            model, optim, criterion = update_output_layer(K = K, label_one_hot = label_one_hot, batch_size = Config.batch_size, model_feature = model_feature, features=features, lr=Config.lr, verbose=False)

    ### CNN Training ### 
        print('Start CNN training')

        # learning rate decay
        #scheduler  = StepLR(optim, step_size = 1, gamma = 0.1)
        milestones = [65, 90]  # Add the milestones where you want to change the learning rate
        scheduler = MultiStepLR(optim, milestones=milestones, gamma=0.1)

        dataset = Subtomogram_Dataset(x_train_permute, labels_permute)
        
        train_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)  

        model.train()  
        iteration_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        start_time = time.time()

        for epoch in range(10):  # Loop for training epochs
            scheduler.step()  # Update learning rate at milestones
            print(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()}')

        
        pbar = tqdm(train_loader, desc='Iterating over train data, Iteration: {}/{}'.format(i - 1, Config.M))
        for train, label in pbar:

            # pass to device
            train = train.to(Config.device)
            label = label.to(Config.device)

            pred = model(train)

            optim.zero_grad()

            label_1 = torch.argmax(label, 1)

            loss = criterion(pred, label_1)           
            iteration_loss += loss.item()

            loss.backward()
            optim.step()

            predicted = torch.argmax(pred, 1)
            train_correct += (predicted == label_1).sum().float().item()
            train_total += label.size(0)
            # lr = optim.param_groups[0]['lr']
            # print('lr', lr)
        scheduler.step()
        # lr = optim.param_groups[0]['lr']
        # print('lr', lr)
        total_loss.append(iteration_loss)

        # calculate accuracy
        accuracy = train_correct / train_total

        exec_time = time.time() - start_time
        print('Loss: {:.4f} Accuracy: {:.4f}  In: {:.4f}s'.format(iteration_loss, accuracy, exec_time))
        # print('lr: {:.4f} Loss: {:.4f} Accuracy: {:.4f}  In: {:.4f}s'.format(lr, iteration_loss, accuracy, exec_time))

        # Inside your loop
        iterations.append(i)
        losses.append(iteration_loss)
        accuracies.append(accuracy)
        execution_times.append(exec_time)
        learning_rate.append(scheduler.get_last_lr()[0])

        if K == 5:   ### This is for evaluating accuracy on simulated data        
            labels_gt = align_cluster_index(gt, labels) 
 
            print('Accuracy:', np.sum(labels_gt == gt)/len(gt), '############################################') 
 
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(gt, labels)                                                             
                                                      
        print('Homogeneity score:', homogeneity, '############################################')                                                          
        print('Completeness score:', completeness, '############################################')                                              
        print('V_measure:', v_measure, '############################################')  

                                                                  
    data_name = os.path.splitext(os.path.basename(path_v))[0]
    data_name.replace('_v','')
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, marker='o')
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    #plt.savefig(f'/home/mohamad.kassab/aitom/result/{i}_{data_name}_loss_plot.png')  # Save the plot as an image
    plt.close()  # Close the plot to release memory

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, marker='o', color='green')
    plt.title('Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    #plt.savefig(f'/home/mohamad.kassab/aitom/result/{i}_{data_name}_accuracy_plot.png')  # Save the plot as an image
    plt.close()  # Close the plot to release memory

    # Plot Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, learning_rate, marker='o', color='orange')
    plt.title('Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    #plt.savefig(f'/home/mohamad.kassab/aitom/result/{i}_{data_name}_learning_rate_plot.png')  # Save the plot as an image
    plt.close()  # Close the plot to release memory

    #model_path_test = f'/home/mohamad.kassab/aitom/model/{i}_{data_name}_model_anythin_torch.pth'  ### path for saving torch model, should be a pth file ###
    #label_path_test = f'/home/mohamad.kassab/aitom/model/{i}_{data_name}_label_anthing_path_torch.pickle'  ### path for saving labels, should be a .pickle file ###
    #torch.save(model, model_path_test)  ### save model here ###
    #print(f'{i}_Model Saved to: {model_path_test}')
    #pickle_dump(labels, label_path_test)

    # # Plot Execution Time
    # plt.figure(figsize=(10, 6))
    # plt.plot(iterations, execution_times, marker='o', color='orange')
    # plt.title('Execution Time vs Iteration')
    # plt.xlabel('Iteration')
    # plt.ylabel('Execution Time (s)')
    # plt.grid(True)
    # plt.savefig('/home/mohamad.kassab/aitom/result/_{data_name}_execution_time_plot.png')  # Save the plot as an image
    # plt.close()  # Close the plot to release memory