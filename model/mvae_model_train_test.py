# %%
logs_path = './logs/mvae_b1k_e80k_eval'

# %% [markdown]
# **This file is to train the model and use the trained to test and visualize the performance of prediction (reconstruction).**

# %%
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io
import math
import sys
import pandas as pd 

import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# %%
### load the augmented dataset, make sure the path to the dataset is correct (by default is on the same directory) 
print("Loading dataset...")

df_training_data = pd.read_csv('/home/stella/Revised_James_Code/final_project_mVAE_pipeline/data_processing/final_aug_training_data.csv', header=None, skiprows=1, index_col=[0]).reset_index(drop=True)
# df_training_data = pd.read_csv('/home/jialuyu/Data_Final_Project/Revised_James_Code/final_project_mVAE_pipeline/data_processing/final_aug_training_data.csv', header=None, skiprows=1, index_col=[0]).reset_index(drop=True)
# df_training_data = df_training_data.iloc[:10000]
X_augm_train = df_training_data.to_numpy()

print(X_augm_train.shape)
n_samples = X_augm_train.shape[0]

# %%
import numpy as np

print("Checking for missing values...")
print(np.isnan(X_augm_train).any(), np.isinf(X_augm_train).any())

nan_columns = np.any(np.isnan(X_augm_train), axis=0)
inf_columns = np.any(np.isinf(X_augm_train), axis=0)

nan_rows = np.where(np.isnan(X_augm_train).any(axis=1))[0]
print(f"Rows with NaN values: {nan_rows}")
print(f"Total rows with NaN values: {len(nan_rows)}")


print(f"Columns with NaN values: {np.where(nan_columns)[0]}")
print(f"Columns with Inf values: {np.where(inf_columns)[0]}")

if np.any(np.isnan(df_training_data)):
    print("Dataset contains NaN values.")
else:
    print("No NaN values detected.")

if np.any(np.isinf(df_training_data)):
    print("Dataset contains Inf values.")
else:
    print("No Inf values detected.")
X_augm_train = X_augm_train[~np.isnan(X_augm_train).any(axis=1)]
print(f"Cleaned dataset shape: {X_augm_train.shape}")


# %%
### train_final_completeloss.py -- core part
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(1.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(1.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),  minval=low, maxval=high,  dtype=tf.float64)

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian distributions and  
    realized by multi-layer perceptrons. The VAE can be learned end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self,sess, network_architecture, transfer_fct=tf.nn.relu,  learning_rate=0.001, batch_size=100, vae_mode=False, vae_mode_modalities=False):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vae_mode = vae_mode
        self.vae_mode_modalities = vae_mode_modalities

        self.n_mc = 4
        self.n_vis = 4

        self.n_input   = network_architecture['n_input']
        self.n_z  = network_architecture['n_z']

        self.x   = tf.placeholder(tf.float64, [None, self.n_input],   name='InputData')
        self.x_noiseless   = tf.placeholder(tf.float64, [None, self.n_input],   name='NoiselessData')
        
        self.layers={}

        self.n_epoch = tf.zeros([],tf.float64)

        # Create autoencoder network
        self._create_network()

        # Define loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()
       
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer() #tf.initialize_all_variables() # 

        # Launch the session
        self.sess = sess #tf.InteractiveSession()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        
        # Summary monitors
        tf.summary.scalar("loss",self.cost) #tf.summary.FileWriter(logs_path) #
        # tf.summary.scalar("loss_J",self.cost_J)
        self.merged_summary_op = tf.summary.merge_all() #tf.merge_all_summaries()

    def _slice_input(self, input_layer, size_mod):
        slices=[]
        count =0
        for i in range(len(self.network_architecture[size_mod])):
            assert self.batch_size % self.network_architecture['size_slices'][i] == 0, \
            f"Batch size {self.batch_size} is not compatible with slice size {self.network_architecture['size_slices'][i]}"

            new_slice = tf.slice(input_layer, [0,count], [self.batch_size,self.network_architecture[size_mod][i]]) # tf.slice(layer_2, [0,200], [105,100])
            count+=self.network_architecture[size_mod][i]
            slices.append(new_slice)

        return slices

    def _create_partial_network(self,name,input_layer):
        with tf.name_scope(name):
            self.layers[name]=[input_layer]
            for i in range(len(self.network_architecture[name])):
                h=tf.Variable(xavier_init(int(self.layers[name][-1].get_shape()[1]), self.network_architecture[name][i]))
                b= tf.Variable(tf.zeros([self.network_architecture[name][i]], dtype=tf.float64))
                layer = self.transfer_fct(tf.add(tf.matmul(self.layers[name][-1],    h), b))
                self.layers[name].append(layer)
            
    def _create_variational_network(self, input_layer, latent_size):
        input_layer_size= int(input_layer.get_shape()[1])
        
        h_mean= tf.Variable(xavier_init(input_layer_size, latent_size))
        h_var= tf.Variable(xavier_init(input_layer_size, latent_size))
        b_mean= tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        b_var= tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        mean = tf.add(tf.matmul(input_layer, h_mean), b_mean)
        log_sigma_sq = tf.log(tf.exp(tf.add(tf.matmul(input_layer, h_var), b_var)) + 0.0001 )
        return mean, log_sigma_sq

    def _create_modalities_network(self, names, slices):
        for i in range(len(names)):
            self._create_partial_network(names[i],slices[i])

    def _create_mod_variational_networ(self, names, sizes_mod):
                assert len(self.network_architecture[sizes_mod])==len(names)
                sizes=self.network_architecture[sizes_mod]
                self.layers['final_means']=[]
                self.layers['final_sigmas']=[]
                for i in range(len(names)):
                        mean, log_sigma_sq=self._create_variational_network(self.layers[names[i]][-1],sizes[i])
                        self.layers['final_means'].append(mean)
                        self.layers['final_sigmas'].append(log_sigma_sq)
                global_mean=tf.concat(self.layers['final_means'],1)
                global_sigma=tf.concat(self.layers['final_sigmas'],1)
                self.layers["global_mean_reconstr"]=[global_mean]
                self.layers["global_sigma_reconstr"]=[global_sigma]
                return global_mean, global_sigma
                                       
    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.x_noiseless_sliced=self._slice_input(self.x_noiseless, 'size_slices')
        slices=self._slice_input(self.x, 'size_slices')
        self._create_modalities_network(['mod0','mod1','mod2','mod3','mod4', 'mod5'], slices)

        self.output_mod = tf.concat([self.layers['mod0'][-1],self.layers['mod1'][-1],self.layers['mod2'][-1],self.layers['mod3'][-1],self.layers['mod4'][-1],self.layers['mod5'][-1]],1) 
        self.layers['concat']=[self.output_mod]
        
        #self._create_partial_network('enc_shared',self.x)
        self._create_partial_network('enc_shared',self.output_mod)
        self.z_mean, self.z_log_sigma_sq = self._create_variational_network(self.layers['enc_shared'][-1],self.n_z)

        if self.vae_mode:
                eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float64)
                self.z   = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        else:
                self.z   = self.z_mean
        

        self._create_partial_network('dec_shared',self.z)

        slices_shared=self._slice_input(self.layers['dec_shared'][-1], 'size_slices_shared')
        self._create_modalities_network(['mod0_2','mod1_2','mod2_2','mod3_2','mod4_2','mod5_2'], slices_shared)

        self.x_reconstr, self.x_log_sigma_sq = self._create_mod_variational_networ(['mod0_2','mod1_2','mod2_2','mod3_2','mod4_2','mod5_2'],'size_slices')
                                                                                                                     
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        with tf.name_scope('Loss_Opt'):
                        self.alpha = 1- tf.minimum(self.n_epoch/1000, 1) # the coefficent used to reduce the impact of latent loss
                        # self.alpha = 1 - tf.minimum(self.n_epoch / 10000, 1)

                        self.tmp_costs=[]
                        for i in range(len(self.layers['final_means'])):
                                reconstr_loss = ( 0.5 * tf.reduce_sum(tf.square(self.x_noiseless_sliced[i] - self.layers['final_means'][i]) / tf.exp(self.layers['final_sigmas'][i] + 1e-6),1) \
                                                + 0.5 * tf.reduce_sum(self.layers['final_sigmas'][i],1) \
                                                + 0.5 * self.n_z/2 * np.log(2*math.pi) )/self.network_architecture['size_slices'][i]
                                self.tmp_costs.append(reconstr_loss)
                                
                        self.reconstr_loss = tf.reduce_mean(self.tmp_costs[0]+ self.tmp_costs[1] + self.tmp_costs[2] + self.tmp_costs[3] + self.tmp_costs[4] + self.tmp_costs[5])

                        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq  - tf.square(self.z_mean)  - tf.exp(self.z_log_sigma_sq), 1)

                        self.cost = tf.reduce_mean(self.reconstr_loss + tf.scalar_mul( self.alpha, self.latent_loss))  # average over batch

                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)  # Use ADAM optimizer

                        self.m_reconstr_loss = self.reconstr_loss
                        self.m_latent_loss = tf.reduce_mean(self.latent_loss)         

    def print_layers_size(self):
        print(self.cost)
        for layer in self.layers:
            print(layer)
            for l in self.layers[layer]:
                print(l)

    def partial_fit(self,sess, X, X_noiseless, epoch):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """

        opt, cost, recon, latent, x_rec, alpha = sess.run((self.optimizer, self.cost, self.m_reconstr_loss,self.m_latent_loss, self.x_reconstr, self.alpha), 
            feed_dict={self.x: X, self.x_noiseless: X_noiseless, self.n_epoch: epoch})
        return cost, recon, latent, x_rec, alpha

    def transform(self,sess, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        return sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self,sess, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is generated. Otherwise, z_mu is drawn from prior in latent space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.n_z)
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        return sess.run(self.x_reconstr, feed_dict={self.z: z_mu})
    
    def reconstruct(self,sess, X_test):
        """ Use VAE to reconstruct given data. """
        x_rec_mean,x_rec_log_sigma_sq = sess.run((self.x_reconstr, self.x_log_sigma_sq), 
            feed_dict={self.x: X_test})
        return x_rec_mean,x_rec_log_sigma_sq


def shuffle_data(x):
    y = list(x)
    np.random.shuffle(y)
    return np.asarray(y)

def train_whole(sess,vae, input_data, learning_rate=0.0001, batch_size=100, training_epochs=10, display_step=10, vae_mode=True, vae_mode_modalities=True):
    print('display_step:' + str(display_step))
    epoch_list = []
    avg_cost_list = []
    avg_recon_list = []    
    avg_latent_list = []

    # Write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
  
    # Training cycle for whole network
    for epoch in tqdm(range(training_epochs)):
        avg_cost = 0.
        avg_recon = 0.
        avg_latent = 0.
        total_batch = int(n_samples / batch_size) 

        X_shuffled = shuffle_data(input_data)

        # Loop over all batches
        for i in range(total_batch):

            batch_xs_augmented = X_shuffled[batch_size*i:batch_size*i+batch_size] 
            
            batch_xs   = np.asarray([item[:108]   for item in batch_xs_augmented]) # augmented (masked) data
            batch_xs_noiseless   = np.asarray([item[108:]   for item in batch_xs_augmented])  # target data
            
            # Fit training using batch data
            cost, recon, latent, x_rec, alpha = vae.partial_fit(sess, batch_xs, batch_xs_noiseless,epoch)
            avg_cost += cost / n_samples * batch_size
            avg_recon += recon / n_samples * batch_size
            avg_latent += latent / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            epoch_list.append(epoch)
            avg_cost_list.append(avg_cost)
            avg_recon_list.append(avg_recon)
            avg_latent_list.append(avg_latent)
        if epoch % 1000 == 0:
            print("Epoch: %04d / %04d, Cost= %04f, Recon= %04f, Latent= %04f, alpha= %04f" % \
                (epoch,training_epochs,avg_cost, avg_recon, avg_latent, alpha))
                        
    ### Save the trained model
    param_id= 1
        # Define the path where the model will be saved
    save_path = "./models_e80000_final/b1k_e80k_eval/mvae_conf_" + str(param_id) + ".ckpt"

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model checkpoint
    vae.saver.save(vae.sess, save_path)
    # save_path = vae.saver.save(vae.sess, "./model/b1k_e80k/mvae_conf_"+str(param_id)+".ckpt")
    return epoch_list, avg_cost_list, avg_recon_list, avg_latent_list


def network_param():
    ### Network parameters are adapted to suit my project, details explained in Model section of the github repo
    network_architecture = \
                            {'n_input':108,\
                              'n_z':100,\
                              'size_slices':[20, 16, 20, 16, 18, 18],\
                              # BECAREFUL The number of slice should be equal to the number of _mod_ network
                              'size_slices_shared':[50, 40, 50, 40, 45, 45],\
                              # BECAREFUL The sum of the dimensions of the slices, should be equal to the last dec_shared
                              'mod0':[100,50],\
                              'mod1':[80,40],\
                              'mod2':[100,50],\
                              'mod3':[80,40],\
                              'mod4':[90,45],\
                              'mod5':[90,45],\
                              'mod0_2':[100,20],\
                              'mod1_2':[80,16],\
                              'mod2_2':[100,20],\
                              'mod3_2':[80,16],\
                              'mod4_2':[90,18],\
                              'mod5_2':[90,18],\
                              'enc_shared':[350],\
                              'dec_shared':[350,270]}
    print(f"Size slices sum: {sum(network_architecture['size_slices'])}")
    print(f"Expected n_input: {network_architecture['n_input']}")
    print(f"Size slices shared sum: {sum(network_architecture['size_slices_shared'])}")
    print(f"Decoder output size: {network_architecture['dec_shared'][-1]}")
    return network_architecture


# %%
## Set training parameters
learning_rate = 0.00005
batch_size = 1440

# Train Network
print('Train net')

sess = tf.InteractiveSession()

vae_mode=True
vae_mode_modalities=False

reload_modalities=False
reload_shared=False
  
vae = VariationalAutoencoder(sess,network_param(),  learning_rate=learning_rate,  batch_size=batch_size, vae_mode=vae_mode, vae_mode_modalities=vae_mode_modalities)
vae.print_layers_size()

epoch_list, avg_cost_list, avg_recon_list, avg_latent_list = train_whole(sess,vae, X_augm_train, training_epochs=80000,batch_size=batch_size)

# %% [markdown]
# 

# %%
#save the stats during training 
np.savetxt('avg_cost_list_models_e80000_final.csv', avg_cost_list)
np.savetxt('avg_recon_list_models_e80000_final.csv', avg_recon_list)
np.savetxt('avg_latent_list_models_e80000_final.csv', avg_latent_list)

# %%
# plot avg cost during training epoches
plt.figure(figsize=(10, 5))
plt.plot(epoch_list, avg_cost_list, label = "avg_cost")
plt.legend()
plt.show()
plt.savefig('avg_cost_models_e80000_final.png', dpi=300, bbox_inches='tight')

# %%
# plot avg reconstruction loss during training epoches
plt.figure(figsize=(10, 5))
plt.plot(epoch_list, avg_recon_list, label = "avg_recon")
plt.legend()
plt.show()
plt.savefig('avg_recon_models_e80000_final.png', dpi=300, bbox_inches='tight')

# %%
# plot avg latent loss during training epoches
plt.figure(figsize=(10, 5))
plt.plot(epoch_list, avg_latent_list, label = "avg_latent")
plt.legend()
plt.show()
plt.savefig('avg_latent_models_e80000_final.png', dpi=300, bbox_inches='tight')

# %%
np.std(avg_cost_list)

# %%
np.std(avg_recon_list)

# %%
np.std(avg_latent_list)

# %%
##load original dataset as ground true value
df_testing_data = pd.read_csv('./data_processing/original_data_with_cur_prev.csv', header=None, skiprows=1, index_col=[0]).reset_index(drop=True)
X_augm_test = df_testing_data.to_numpy()
print(X_augm_test.shape)
n_samples = X_augm_test.shape[0]

# %%
################################################################################################################
## Using the trained model to test the perforamnce of prediction (recontrsuction)
with tf.Graph().as_default() as g:
  with tf.Session() as sess:

      # Network parameters
      network_architecture = network_param()
      learning_rate = 0.00001
      batch_size = 1440 # use the task one datapoints
      sample_init = 0

      model = VariationalAutoencoder(sess,network_architecture, batch_size=batch_size, learning_rate=learning_rate, vae_mode=False, vae_mode_modalities=False)

  with tf.Session() as sess:
      new_saver = tf.train.Saver()
      param_id= 1
      new_saver.restore(sess, "./models_e70000/b1k_e80k_eval/mvae_conf_"+str(param_id)+".ckpt") ###load trained model
      print("Model restored.")
                        
      ################################################################################################################
      # Test 1: complete data
      print('Test 1')
      x_sample = X_augm_test[sample_init:sample_init+batch_size,:108]  
      x_reconstruct_1, x_reconstruct_log_sigma_sq_1 = model.reconstruct(sess,x_sample)

      ################################################################################################################
      #Test 2: human data only
      print('Test 2')
      x_sample_nv_1 = X_augm_test[sample_init:sample_init+batch_size,:72]  
      x_sample_nv_2 = np.full((x_sample.shape[0],36),-2)  
      x_sample_nv = np.append( x_sample_nv_1, x_sample_nv_2, axis=1)
                
      x_reconstruct_2, x_reconstruct_log_sigma_sq_2 = model.reconstruct(sess,x_sample_nv) 

      ################################################################################################################      
      # Test 3: only data at time t -- for prediction
      print('Test 3')
      x_sample_nv_1 = np.full((x_sample.shape[0],10),-2)  
      x_sample_nv_2 = X_augm_test[sample_init:sample_init+batch_size,10:20] # qt-1
      x_sample_nv_3 = np.full((x_sample.shape[0],8),-2)  
      x_sample_nv_4 = X_augm_test[sample_init:sample_init+batch_size,28:36] 
      x_sample_nv_5 = np.full((x_sample.shape[0],10),-2)  
      x_sample_nv_6 = X_augm_test[sample_init:sample_init+batch_size,46:56] 
      x_sample_nv_7 = np.full((x_sample.shape[0],8),-2)  
      x_sample_nv_8 = X_augm_test[sample_init:sample_init+batch_size,64:72]
      x_sample_nv_9 = np.full((x_sample.shape[0],9),-2)
      x_sample_nv_10 = X_augm_test[sample_init:sample_init+batch_size,81:90] 
      x_sample_nv_11 = np.full((x_sample.shape[0],9),-2)
      x_sample_nv_12 = X_augm_test[sample_init:sample_init+batch_size,99:108] 
      x_sample_nv = np.append(x_sample_nv_1,
                                        np.append(x_sample_nv_2,
                                                  np.append(x_sample_nv_3,
                                                            np.append(x_sample_nv_4,
                                                                      np.append(x_sample_nv_5,
                                                                                np.append(x_sample_nv_6,
                                                                                          np.append(x_sample_nv_7,
                                                                                                    np.append(x_sample_nv_8,
                                                                                                              np.append(x_sample_nv_9,
                                                                                                                        np.append(x_sample_nv_10,
                                                                                                                                  np.append(x_sample_nv_11,
                                                                                                                                            x_sample_nv_12,axis=1)
                                                                                                              ,axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1)

      x_reconstruct_3, x_reconstruct_log_sigma_sq_3 = model.reconstruct(sess,x_sample_nv) 
      ################################################################################################################
      # Test 4: use VAE for prediction: first reconstruct from human only, then reconstruct again from reconstructed data
      print('USING VAE FOR PREDICTIONS')
      x_sample_nv_1 = X_augm_test[sample_init:sample_init+batch_size,:72]  
      x_sample_nv_2 = np.full((x_sample.shape[0],36),-2)  
      x_sample_nv = np.append( x_sample_nv_1, x_sample_nv_2, axis=1)

      x_reconstruct_4, x_reconstruct_log_sigma_sq_4 = model.reconstruct(sess,x_sample_nv)
      # Using the reconstructed values (at t) again (from previous step) to predict the future (at t+1) values
      x_sample_nv_1 = np.full((x_sample.shape[0],10),-2)  
      x_sample_nv_2 = x_reconstruct_4[:,:10]  
      x_sample_nv_3 = np.full((x_sample.shape[0],8),-2)  
      x_sample_nv_4 = x_reconstruct_4[:,20:28]  
      x_sample_nv_5 = np.full((x_sample.shape[0],10),-2)  
      x_sample_nv_6 = x_reconstruct_4[:,36:46]  
      x_sample_nv_7 = np.full((x_sample.shape[0],8),-2)  
      x_sample_nv_8 = x_reconstruct_4[:,56:64]  
      x_sample_nv_9 = np.full((x_sample.shape[0],9),-2)
      x_sample_nv_10 = x_reconstruct_4[:,72:81] 
      x_sample_nv_11 = np.full((x_sample.shape[0],9),-2)
      x_sample_nv_12 = x_reconstruct_4[:,90:99] 
      x_sample_nv = np.append(x_sample_nv_1,
                                        np.append(x_sample_nv_2,
                                                  np.append(x_sample_nv_3,
                                                            np.append(x_sample_nv_4,
                                                                      np.append(x_sample_nv_5,
                                                                                np.append(x_sample_nv_6,
                                                                                          np.append(x_sample_nv_7,
                                                                                                    np.append(x_sample_nv_8,
                                                                                                              np.append(x_sample_nv_9,
                                                                                                                        np.append(x_sample_nv_10,
                                                                                                                                  np.append(x_sample_nv_11,
                                                                                                                                            x_sample_nv_12,axis=1)
                                                                                                              ,axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1),axis=1)
      x_pred, x_pred_log_sigma_sq = model.reconstruct(sess,x_sample_nv) 

# %%
from sklearn.metrics import mean_squared_error
print('test1')
pos_true = x_sample[:,72:81]
pos_pred = x_reconstruct_1[:,72:81]
print(mean_squared_error(pos_true, pos_pred))
vel_true = x_sample[:,90:99]
vel_pred = x_reconstruct_1[:,90:99]
print(mean_squared_error(vel_true, vel_pred))


# %%
print('test2')
pos_true = x_sample[:,72:81]
pos_pred = x_reconstruct_2[:,72:81]
print(mean_squared_error(pos_true, pos_pred))
vel_true = x_sample[:,90:99]
vel_pred = x_reconstruct_2[:,90:99]
print(mean_squared_error(vel_true, vel_pred))

# %%
print('test3')
pos_true = x_sample[:,72:81]
pos_pred = x_reconstruct_3[:,72:81]
print(mean_squared_error(pos_true, pos_pred))
vel_true = x_sample[:,90:99]
vel_pred = x_reconstruct_3[:,90:99]
print(mean_squared_error(vel_true, vel_pred))

# %%
print('test4: prediction')
pos_true = x_sample[1:,72:81]
pos_pred = x_pred[:-1,72:81]
print(mean_squared_error(pos_true, pos_pred))
vel_true = x_sample[1:,90:99]
vel_pred = x_pred[:-1,90:99]
print(mean_squared_error(vel_true, vel_pred))

# %%
## Example plots for task one: 
# Comparing the prediced values (from test2) and original values for the positions and velocities of 7 joints 
fig, ax = plt.subplots(9,2,figsize=(20, 15))
x = [i for i in range(len(x_sample))]
j_pos_true = x_sample[:,72:81]
j_vel_true = x_sample[:,90:99]
j_pos_pred = x_reconstruct_2[:,72:81]
j_vel_pred = x_reconstruct_2[:,90:99]

for i in range(9):
  ax[i,0].plot(x, j_pos_true[:,i], label='original pos')
  ax[i,0].plot(x, j_pos_pred[:,i], label='pred pos')

  ax[i,1].plot(x, j_vel_true[:,i], label='original vel')
  ax[i,1].plot(x, j_vel_pred[:,i], label='pred vel')

ax[0,0].set_title('Joints Positions Comparing')
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')
ax[0,1].set_title('Joints Velocities Comparing')
handles, labels = ax[0,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')  
plt.tight_layout()
plt.savefig('p1_models_e80000_final.png', dpi=300, bbox_inches='tight')

# %%
## Example plots for task one: 
# Comparing the prediced values (from test4) and original values for the positions and velocities of 7 joints 
fig, ax = plt.subplots(9,2,figsize=(20, 15))
x = [i for i in range(len(x_sample)-1)]
j_pos_true = x_sample[1:,72:81] # one datapoint shift in original dataset at time t
j_vel_true = x_sample[1:,90:99]
j_pos_pred = x_pred[:-1,72:81] # data at time t + 1
j_vel_pred = x_pred[:-1,90:99]

for i in range(9):
  ax[i,0].plot(x, j_pos_true[:,i], label='original pos')
  ax[i,0].plot(x, j_pos_pred[:,i], label='pred pos')

  ax[i,1].plot(x, j_vel_true[:,i], label='original vel')
  ax[i,1].plot(x, j_vel_pred[:,i], label='pred vel')

ax[0,0].set_title('Joints Positions Comparing')
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left')
ax[0,1].set_title('Joints Velocities Comparing')
handles, labels = ax[0,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')  
plt.tight_layout()
plt.savefig('p2_models_e80000_final.png', dpi=300, bbox_inches='tight')


