import ray
import csv
import sys
import time
import glob
import obspy
import shutil
import platform
import logging
import numpy as np
from os import listdir
from os.path import join
from io import StringIO
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
#tf.config.optimizer.set_jit(True)
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(8)
import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

 
def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+tf.keras.backend.epsilon()))


def wbceEdit( y_true, y_pred) :
    ms = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred)) 
    ssim = 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))
    return (ssim + ms)

w1 = 6000
w2 = 3
drop_rate = 0.2
stochastic_depth_rate = 0.1

positional_emb = False
conv_layers = 4
num_classes = 1
input_shape = (w1, w2)
num_classes = 1
input_shape = (6000, 3)
image_size = 6000  # We'll resize input images to this size
patch_size = 40  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size)
projection_dim = 40

num_heads = 4
transformer_units = [
    projection_dim,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = tf.keras.layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded
    
# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    


def convF1(inpt, D1, fil_ord, Dr):

    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    #filters = inpt._keras_shape[channel_axis]
    filters = int(inpt.shape[-1])
    
    #infx = tf.keras.layers.Activation(tf.nn.gelu')(inpt)
    pre = tf.keras.layers.Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inpt)
    pre = tf.keras.layers.BatchNormalization()(pre)    
    pre = tf.keras.layers.Activation(tf.nn.gelu)(pre)
    
    #shared_conv = tf.keras.layers.Conv1D(D1,  fil_ord, strides =(1), padding='same')
    
    inf = tf.keras.layers.Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(pre)
    inf = tf.keras.layers.BatchNormalization()(inf)    
    inf = tf.keras.layers.Activation(tf.nn.gelu)(inf)
    inf = tf.keras.layers.Add()([inf,inpt])
    
    inf1 = tf.keras.layers.Conv1D(D1,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inf)
    inf1 = tf.keras.layers.BatchNormalization()(inf1)  
    inf1 = tf.keras.layers.Activation(tf.nn.gelu)(inf1)    
    encode = tf.keras.layers.Dropout(Dr)(inf1)

    return encode


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        #x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = tf.keras.layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded
    

def create_cct_model(inputs):

    inputs1 = convF1(inputs,  10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = tf.keras.layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def create_cct_modelP(inputs):

    inputs1 = convF1(inputs,  10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = tf.keras.layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        #encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        #attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def create_cct_modelS(inputs):

    inputs1 = convF1(inputs,  10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = tf.keras.layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def tf_environ(gpu_id, gpu_limit=None):
    if gpu_id != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[{datetime.now()}] GPU processing enabled.")
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        #             if gpu_limit:
        #                 tf.config.experimental.set_virtual_device_configuration(gpu,
        #                                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_limit*1024)])
        #         print(f"[{datetime.now()}] GPU processing enabled.")
        #     except RuntimeError as e:
        #         print(e)
        # else:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        #     print(f"[{datetime.now()}] No GPUs found, using CPU.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(f"[{datetime.now()}] GPU processing disabled, using CPU.")



def load_eqcct_model(input_modelP, input_modelS):
    # print(f"[{datetime.now()}] Loading EQCCT model.")
    
    # with open(log_file, mode="w", buffering=1) as log:
    #     log.write(f"*** Loading the model ...\n")

    # Model CCT
    inputs = tf.keras.layers.Input(shape=input_shape,name='input')

    #featuresP = create_cct_model(inputs)
    #featuresS = create_cct_model(inputs)
    #logitp  = Dense(6000 ,activation='sigmoid', kernel_initializer='he_normal',name='picker_P1')(featuresP)
    #logits  = Dense(6000 ,activation='sigmoid', kernel_initializer='he_normal',name='picker_S1')(featuresS)
    #logitp = tf.keras.layers.Reshape((6000,1), name='picker_P')(logitp)
    #logits = tf.keras.layers.Reshape((6000,1), name='picker_S')(logits)

    featuresP = create_cct_modelP(inputs)
    featuresP = tf.keras.layers.Reshape((6000,1))(featuresP)

    featuresS = create_cct_modelS(inputs)
    featuresS = tf.keras.layers.Reshape((6000,1))(featuresS)

    logitp  = tf.keras.layers.Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_P')(featuresP)
    logits  = tf.keras.layers.Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_S')(featuresS)

    modelP = tf.keras.models.Model(inputs=[inputs], outputs=[logitp])
    modelS = tf.keras.models.Model(inputs=[inputs], outputs=[logits])

    model = tf.keras.models.Model(inputs=[inputs], outputs=[logitp,logits])

    summary_output = StringIO()
    with redirect_stdout(summary_output):
        model.summary()
    # log.write(summary_output.getvalue())
    # log.write('\n')

    sgd = tf.keras.optimizers.Adam()
    model.compile(optimizer=sgd,
                loss=['binary_crossentropy','binary_crossentropy'],
                metrics=['acc',f1,precision, recall])    
    
    modelP.load_weights(input_modelP)
    modelS.load_weights(input_modelS)

    # log.write(f"*** Loading is complete!")

    return model
        
def mseed_predictor(stream=None,
              output_dir="detections",
              stations2use=None,
              P_threshold=0.1,
              S_threshold=0.1, 
              normalization_mode='std',
              overlap = 0.3,
              batch_size=500,
              overwrite=False,
              log_file="./results/logs/picker/eqcct.log",           
            #   model=None,
              p_model = None,
              s_model = None,
              numCPUs = 1,
              gpu_id=None,
              gpu_limit=None,
              maxStsTasksQueue=1): 
    
    """ 
    
    To perform fast detection directly on input data.
    
    Parameters
    ----------
    tr: object
        Trace Obspy object obspy.core.stream.
            
    input_model: str
        Path to a trained model.
            
    stations_json: str
        Path to a JSON file containing station information. 
           
    output_dir: str
        Output directory that will be generated.
            
    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.                
            
    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
            
    normalization_mode: str, default=std
        Mode of normalization for data preprocessing max maximum amplitude among three components std standard deviation.
             
    batch_size: int, default=500
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommended.
             
    overlap: float, default=0.3
        If set the detection and picking are performed in overlapping windows.
             
    gpu_id: int
        Id of GPU used for the prediction. If using CPU set to None.        
             
    gpu_limit: float
       Set the maximum percentage of memory usage for the GPU. 

    overwrite: Bolean, default=False
        Overwrite your results automatically.
           
    Returns
    --------        
      
    """ 
 
    args = {
    "output_dir": output_dir,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "normalization_mode": normalization_mode,
    "overlap": overlap,
    "batch_size": batch_size,    
    "gpu_id": gpu_id,
    "gpu_limit": gpu_limit,
    "p_model":p_model,
    "s_model":s_model
    }

    picks = []
    out_dir = os.path.join(os.getcwd(), str(args['output_dir']))

    with open(log_file, mode="w", buffering=1) as log:
        try:
            station_list = stations2use
            station_list = sorted(set(station_list))
        except Exception as exp:
            log.write(f"{exp}")
            return
        log.write(f"[{datetime.now()}] {len(station_list)} stations in the record stream.")
        
        tasks = [[stream.select(network=station_list[i].split('.')[0], station=station_list[i].split('.')[1]), f"({i+1}/{len(station_list)})", station_list[i], args, out_dir] for i in range(len(station_list))]
        if not tasks:
            return
        #parallel_predict(tasks[0])
        # Submit tasks to ray in a queue
        tasks_queue = []
        log.write(f"[{datetime.now()}] Started EQCCT picking process.\n")
        for i in range(len(tasks)):
            while True:
                # Add new task to queue while max is not reached
                if len(tasks_queue) < maxStsTasksQueue:
                    tasks_queue.append(parallel_predict.remote(tasks[i]))
                    break
                # If there are more tasks than maximum, just process them
                else:
                    tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                    for finished_task in tasks_finished:
                        picks_parall = ray.get(finished_task)
                        picks.append(picks_parall['picks'])
                        log.write(f"{picks_parall['info']}")

        # After adding all the tasks to queue, process what's left
        while tasks_queue:
            tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
            for finished_task in tasks_finished:
                picks_parall = ray.get(finished_task)
                for pick in picks_parall['picks']:
                    picks.append(pick)
                log.write(f"{picks_parall['info']}")
        ray.shutdown()

        # # with ThreadPoolExecutor(max_workers=min([round(os.cpu_count()*0.25), len(tasks)])) as executor:
        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     futures = [executor.submit(parallel_predict, task) for task in tasks]
        #     for future in as_completed(futures):
        #         result_dict = future.result()
        #         log_entry = result_dict['info']
        #         picks_parall = result_dict['picks']
        #         picks.extend(picks_parall)
        #         log.write(log_entry + "")
        # log.write("------- END OF FILE -------")
    
    return (picks)

@ray.remote
def parallel_predict(predict_args):
    stream, pos, st, args, out_dir= predict_args
    model = load_eqcct_model(args["p_model"], args["s_model"])
    save_dir = os.path.join(out_dir, str(st)+'_outputs')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    csvPr_gen = open(os.path.join(save_dir,'X_prediction_results.csv'), 'w')     
    predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_writer.writerow(['file_name', 
                            'network',
                            'station',
                            'instrument_type',
                            'station_lat',
                            'station_lon',
                            'station_elv',
                            'p_arrival_time',
                            'p_probability',
                            's_arrival_time',
                            's_probability'])  
    csvPr_gen.flush()
    #print(f"[{datetime.now()}] Started working on {st}...")
    
    start_Predicting = time.time()  
    time_slots, comp_types = [], []
    
    # log.write('============ Station {} has {} chunks of data.'.format(st, len(uni_list)), flush=True)
    #log.write(f"{f}")
    # try:
    meta, time_slots, comp_types, data_set = _readnparray(stream, args, time_slots, comp_types, st)
    # except: #InternalMSEEDError:
        # return f"[{datetime.now()}] {pos} {st}: FAILED the prediction (reading mSEED)."

    # print(st)
    # print(data_set)
    batch_sizes = [args['batch_size'], 500, 250, 100, 50, 10, 5, 1]
    i = 0
    while True:
        # try:
        params_pred = {'batch_size': batch_sizes[i], 'norm_mode': args['normalization_mode']}
        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
        predP,predS = model.predict(pred_generator, verbose=0)
        
        #log.write(np.shape(predP))
        detection_memory = []
        prob_memory=[]
        picks = []
        for ix in range(len(predP)):
            Ppicks, Pprob =  _picker(args, predP[ix,:, 0])   
            Spicks, Sprob =  _picker(args, predS[ix,:, 0], 'S_threshold')
            #log.write(np.shape(Ppicks))

            #### Add option to activate write picks in csv file
            # detection_memory,prob_memory=_output_writter_prediction(meta, csvPr_gen, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, ix,len(predP),len(predS))
            pick,detection_memory,prob_memory=_output_dict_prediction(meta, Ppicks, Pprob, Spicks, Sprob, detection_memory, prob_memory, ix,len(predP),len(predS))
            picks.append(pick)
                                        
        end_Predicting = time.time() 
        #data_track[st]=[time_slots, comp_types] 
        delta = (end_Predicting - start_Predicting) 
        # hour = int(delta / 3600)
        # delta -= hour * 3600
        # minute = int(delta / 60)
        # delta -= minute * 60
        # seconds = delta     
                        
        # dd = pd.read_csv(os.path.join(save_dir,'X_prediction_results.csv'))
        #print(f"[{datetime.now()}] {pos} {st}: Finished the prediction in {round(delta,2)}s.")
        # return f"[{datetime.now()}] {pos} {st}: Finished the prediction in {round(delta,2)}s."
        info = f"[{datetime.now()}] {pos} {st}: Finished the prediction in {round(delta,2)}s."
        return {
            'picks':picks,
            'info':info
        }
        # log.write(f'*** Detected: '+str(len(dd))+' events.')
        #log.write(f'*** Wrote the results into --> {str(save_dir)}')
        # except:
        #     if i >= len(batch_sizes):
        #         return f"[{datetime.now()}] {pos} {st}: FAILED the prediction."
        #     # print(f"[{datetime.now()}] {pos} {st}: Failed the prediction. Retrying in 5s.")
        #     i += 1
        #     time.sleep(5)


def _readnparray(stream, args, time_slots, comp_types, st_name):
    'Read data from record stream and returns 3 dictionaries of numpy arrays, meta data, and time slice info'

    output_wf = f"/home/cam/EQCCT/sceqcct/eqcct-dev/results/waveform"
    # print('st_name')
    # print(st_name)

    st = obspy.Stream()
    st = stream
    st.merge(fill_value=0)
    st.detrend("linear")
    st.detrend('demean')

    # temp_st = obspy.Stream()
    # # temp_st.append(tr)
    # temp_st = stream
    # try:
    #     temp_st.merge(fill_value=0)                     
    # except Exception:
    #     #temp_st =_resampling(temp_st)
    #     temp_st.merge(fill_value=0) 
    # temp_st.detrend('demean')
    # if len(temp_st) > 0:
    #     st += temp_st
    # else:
    #     return

    # print('ANTES DE PROCESAR DATOS')
    # print(st)
    
    for tr in st:
        start_time = tr.stats.starttime
        end_time = tr.stats.endtime
        # print(tr)
        # print(f'Start time {start_time}')
        # print(f'End time {end_time}')

    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime
    staName = st[0].stats.station
    # print(f'Start time {start_time}')
    # print(f'End time {end_time}')

    # st.write(f"{output_wf}/before/{start_time}_{end_time}_{staName}.mseed", format="MSEED")


    st.taper(max_percentage=0.05, type='cosine')
    st.trim(min([tr.stats.starttime for tr in st])-10, max([tr.stats.endtime for tr in st])+10, pad=True, fill_value=0)
    st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
    st.trim(min([tr.stats.starttime for tr in st])+10, max([tr.stats.endtime for tr in st])-10)
    
    st.taper(max_percentage=0.001, type='cosine', max_length=2)
    if len([tr for tr in st if tr.stats.sampling_rate != 100.0]) != 0:
        try:
            st.interpolate(100, method="linear")
        except Exception:
            st=_resampling(st)  
    
    st.trim(min([tr.stats.starttime for tr in st]), max([tr.stats.endtime for tr in st]), pad=True, fill_value=0)

    # st.write(f"{output_wf}/after/{start_time}_{end_time}_{staName}.mseed", format="MSEED")

    meta = {"start_time":start_time,
            "end_time":end_time,
            # "trace_name":f"{f.split('/')[-2]}/{f.split('/')[-1]}"
            "trace_name":f"{st_name}.{st[0].stats.location}.{st[0].stats.channel}"
             } 
    
    chanL = [tr.stats.channel[-1] for tr in st]
    comp_types.append(len(chanL))
    # time_shift = int(60-(args['overlap']*60))
    # next_slice = start_time + 60
    data_set = {}
    st_times = []        
    # sl = 0 
    # print(next_slice)
    # print(end_time)
    # while next_slice <= end_time:

    # tr_size = len(st[chanL.index(chanL[0])])
    tr_size = 6000
    npz_data = np.zeros([tr_size, 3]) 
    st_times.append(str(start_time).replace('T', ' ').replace('Z', ''))

    # w = st.slice(start_time, next_slice)
    w = st
    # print(w)
    # print(chanL)
    if 'Z' in chanL:
        npz_data[:,2] = w[chanL.index('Z')].data[:tr_size]
    if ('E' in chanL) or ('1' in chanL):
        try: 
            npz_data[:,0] = w[chanL.index('E')].data[:tr_size]
        except Exception:
            npz_data[:,0] = w[chanL.index('1')].data[:tr_size]
    if ('N' in chanL) or ('2' in chanL):
        try:
            npz_data[:,1] = w[chanL.index('N')].data[:tr_size]
        except Exception:
            npz_data[:,1] = w[chanL.index('2')].data[:tr_size]
            
    data_set.update({str(start_time).replace('T', ' ').replace('Z', '') : npz_data})
   
        # start_time = start_time + time_shift
        # next_slice = next_slice + time_shift 
        # sl += 1
   
    meta["trace_start_time"] = st_times
    #print(stations_)
    ''' 
    try:
        meta["receiver_code"]=st[0].stats.station
        meta["instrument_type"]=st[0].stats.channel[:2]
        meta["network_code"]=stations_[st[0].stats.station]['network']
        meta["receiver_latitude"]=stations_[st[0].stats.station]['coords'][0]
        meta["receiver_longitude"]=stations_[st[0].stats.station]['coords'][1]
        meta["receiver_elevation_m"]=stations_[st[0].stats.station]['coords'][2]  
    except Exception:
        meta["receiver_code"]=st_name
        meta["instrument_type"]=stations_[st_name]['channels'][0][:2]
        meta["network_code"]=stations_[st_name]['network']
        meta["receiver_latitude"]=stations_[st_name]['coords'][0]
        meta["receiver_longitude"]=stations_[st_name]['coords'][1]
        meta["receiver_elevation_m"]=stations_[st_name]['coords'][2] 
    '''
    try:
        meta["receiver_code"]=st[0].stats.station
        meta["instrument_type"]=0
        meta["network_code"]=0
        meta["receiver_latitude"]=0
        meta["receiver_longitude"]=0
        meta["receiver_elevation_m"]=0
    except Exception:
        meta["receiver_code"]=st_name
        meta["instrument_type"]=0
        meta["network_code"]=0
        meta["receiver_latitude"]=0
        meta["receiver_longitude"]=0
        meta["receiver_elevation_m"]=0
                    
    return meta, time_slots, comp_types, data_set


class PreLoadGeneratorTest(tf.keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. Pre-load version.
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32.
        Batch size.
            
    n_channels: int, default=3.
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'                
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'picker_P': y2, 'picker_S': y3}: outputs including two separate numpy arrays as labels for P, and S respectively.
    
    
    """  
    
    def __init__(self, 
                 list_IDs, 
                 inp_data, 
                 batch_size=32, 
                 norm_mode = 'std',
                 **kwargs):
       
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.inp_data = inp_data        
        self.on_epoch_end()
        self.norm_mode = norm_mode
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        try:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        except ZeroDivisionError:
            print("Your data duration in mseed file is too short! Try either longer files or reducing batch_size. ")
        
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]           
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self.__data_generation(list_IDs_temp)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def _normalize(self, data, mode = 'max'):          
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
                       
    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        X = np.zeros((self.batch_size, 6000, 3))           
        # Generate data
        for i, ID in enumerate(list_IDs_temp):            
            data = self.inp_data[ID]
            data = self._normalize(data, self.norm_mode)                            
            X[i, :, :] = data                                                           
                           
        return X      
    

def _output_writter_prediction(meta, csvPr, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, idx, cq, cqq):

    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    predict_writer: obj
        For writing out the detection/picking results in the CSV file. 
       
    csvPr: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    station_lat = meta["receiver_latitude"]
    station_lon = meta["receiver_longitude"]
    station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    station_name = "{:<4}".format(station_name)
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
    
    p_time = []
    p_prob = []
    PdateTime = []
    if Ppicks[0]!=None: 
#for iP in range(len(Ppicks)):
#if Ppicks[iP]!=None: 
        p_time.append(start_time+timedelta(seconds= Ppicks[0]/100))
        p_prob.append(Pprob[0])
        PdateTime.append(_date_convertor(start_time+timedelta(seconds= Ppicks[0]/100)))
        detection_memory.append(p_time) 
        prob_memory.append(p_prob)  
    else:          
        p_time.append(None)
        p_prob.append(None)
        PdateTime.append(None)

    s_time = []
    s_prob = []    
    SdateTime=[]
    if Spicks[0]!=None: 
#for iS in range(len(Spicks)):
#if Spicks[iS]!=None: 
        s_time.append(start_time+timedelta(seconds= Spicks[0]/100))
        s_prob.append(Sprob[0])
        SdateTime.append(_date_convertor(start_time+timedelta(seconds= Spicks[0]/100)))
    else:
        s_time.append(None)
        s_prob.append(None)
        SdateTime.append(None)

    SdateTime = np.array(SdateTime)
    s_prob = np.array(s_prob)
    
    p_prob = np.array(p_prob)
    PdateTime = np.array(PdateTime)
        
    predict_writer.writerow([meta["trace_name"], 
                     network_name,
                     station_name, 
                     instrument_type,
                     station_lat, 
                     station_lon,
                     station_elv,
                     PdateTime[0], 
                     p_prob[0],
                     SdateTime[0], 
                     s_prob[0]
                     ]) 



    csvPr.flush()                


    return detection_memory,prob_memory  

def _output_dict_prediction(meta, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory, idx, cq, cqq):

    """ 
    
    Writes the detection & picking results into a dictionary.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    station_lat = meta["receiver_latitude"]
    station_lon = meta["receiver_longitude"]
    station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    station_name = "{:<4}".format(station_name)
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
    
    p_time = []
    p_prob = []
    PdateTime = []
    if Ppicks[0]!=None: 
#for iP in range(len(Ppicks)):
#if Ppicks[iP]!=None: 
        p_time.append(start_time+timedelta(seconds= Ppicks[0]/100))
        p_prob.append(Pprob[0])
        PdateTime.append(_date_convertor(start_time+timedelta(seconds= Ppicks[0]/100)))
        detection_memory.append(p_time) 
        prob_memory.append(p_prob)  
    else:          
        p_time.append(None)
        p_prob.append(None)
        PdateTime.append(None)

    s_time = []
    s_prob = []    
    SdateTime=[]
    if Spicks[0]!=None: 
#for iS in range(len(Spicks)):
#if Spicks[iS]!=None: 
        s_time.append(start_time+timedelta(seconds= Spicks[0]/100))
        s_prob.append(Sprob[0])
        SdateTime.append(_date_convertor(start_time+timedelta(seconds= Spicks[0]/100)))
    else:
        s_time.append(None)
        s_prob.append(None)
        SdateTime.append(None)

    SdateTime = np.array(SdateTime)
    s_prob = np.array(s_prob)
    
    p_prob = np.array(p_prob)
    PdateTime = np.array(PdateTime)
    
    pick = {
        "trace_name": meta["trace_name"],
        "network_name": network_name,
        "station_name": station_name,
        "instrument_type": instrument_type,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "station_elv": station_elv,
        "PdateTime": PdateTime[0],
        "p_prob": p_prob[0],
        "SdateTime": SdateTime[0],
        "s_prob": s_prob[0]
    }          

    return pick,detection_memory,prob_memory  


def _get_snr(data, pat, window=200):
    
    """ 
    
    Estimates SNR.
    
    Parameters
    ----------
    data : numpy array
        3 component data.    
        
    pat: positive integer
        Sample point where a specific phase arrives. 
        
    window: positive integer, default=200
        The length of the window for calculating the SNR (in the sample).         
        
    Returns
   --------   
    snr : {float, None}
       Estimated SNR in db. 
       
        
    """      
    import math
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)    
        except Exception:
            pass
    return snr 


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _picker(args, yh3, thr_type='P_threshold'):
    """ 
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
         probability. 

    Returns
    --------    
    Ppickall: Pick.
    Pproball: Pick Probability.                           
                
    """
    P_PICKall=[]
    Ppickall=[]
    Pproball = []
    perrorall=[]

    sP_arr = _detect_peaks(yh3, mph=args[thr_type], mpd=1)

    P_PICKS = []
    pick_errors = []
    if len(sP_arr) > 0:
        P_uncertainty = None  

        for pick in range(len(sP_arr)):        
            sauto = sP_arr[pick]


            if sauto: 
                P_prob = np.round(yh3[int(sauto)], 3) 
                P_PICKS.append([sauto,P_prob, P_uncertainty]) 

    so=[]
    si=[]
    P_PICKS = np.array(P_PICKS)
    P_PICKall.append(P_PICKS)
    for ij in P_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        Ppickall.append((swave))
        Pproball.append((np.max(so)))
    except:
        Ppickall.append(None)
        Pproball.append(None)

    #print(np.shape(Ppickall))
    #Ppickall = np.array(Ppickall)
    #Pproball = np.array(Pproball)
    
    return Ppickall, Pproball


def _resampling(st):
    'perform resampling on Obspy stream objects'
    
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr) 
    return st 


def _normalize(data, mode = 'max'):  
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """  
       
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data
