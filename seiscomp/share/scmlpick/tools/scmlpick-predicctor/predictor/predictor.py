import ray
import csv
import sys, tempfile
import time
import glob
import obspy
import shutil
import platform
import numpy as np
from pathlib import Path
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
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from .eqtransformer import load_eqtransformer_model
from .phasenet import load_phasenet_model
from .utrans_eqcct import load_utrans_eqcct_model

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
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(f"[{datetime.now()}] GPU processing disabled, using CPU.")



def load_eqcct_model(input_modelP, input_modelS):

    # Model CCT
    inputs = tf.keras.layers.Input(shape=input_shape,name='input')

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

    sgd = tf.keras.optimizers.Adam()
    model.compile(optimizer=sgd,
                loss=['binary_crossentropy','binary_crossentropy'],
                metrics=['acc',f1,precision, recall])    
    
    modelP.load_weights(input_modelP)
    modelS.load_weights(input_modelS)

    # log.write(f"*** Loading is complete!")

    return model


# Registry of supported picker architectures. Each entry is a callable with the
# same signature as `load_eqcct_model`: (input_modelP, input_modelS) -> a
# compiled tf.keras.Model whose `.predict(...)` returns (predP, predS), both
# shape (batch, 6000, 1). A new architecture is registered here. PhaseNet keeps
# that output contract, but its loader normalizes each window itself, so the
# generator must not std-normalize those batches first.
MODEL_LOADERS = {
    "eqcct": load_eqcct_model,
    "utrans_eqcct": load_utrans_eqcct_model,
    "phasenet": load_phasenet_model,
    "eqtransformer": load_eqtransformer_model,
}


def mseed_predictor(stream=None,
              filterShift=2.5,
              wfinfo=None,
              files=None,
              P_threshold=0.1,
              S_threshold=0.1,
              normalization_mode='std',
              overlap = 30,
              batch_size=500,
              overwrite=False,
              p_model = None,
              s_model = None,
              numCPUs = 1,
              gpu_id=None,
              gpu_limit=None,
              stations_filters=None,
              architecture='eqcct',
              playback=False,
              model=None,
              precision='bf16',
              detection_threshold=0.3):

    # _LOG.info("mseed_predictor: inicio")

    net, sta, t0, t1 = wfinfo
    net_sta = f"{net}.{sta}"

    args = {
    "filterShift": filterShift,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "normalization_mode": normalization_mode,
    "overlap": overlap,
    "batch_size": batch_size,
    "gpu_id": gpu_id,
    "gpu_limit": gpu_limit,
    "p_model":p_model,
    "s_model":s_model,
    "architecture": architecture,
    "stations_filters": stations_filters,
    "station": net_sta,
    "files":files,
    "t_ini": t0,
    "t_end":t1,
    "playback":playback,
    "detection_threshold": detection_threshold,
    }

    if playback:
        task = [stream,net_sta,args]
    else:
        task = [stream.select(network=net, station=sta), net_sta, args]
    picks = parallel_predict(task, model=model, precision=precision)
    return (picks)

# PhaseNet and UTrans put a false P on the left edge of every 60 s tensor.
# PhaseNet's P softmax spikes there because padding="same" fills the edge
# with zeros. UTrans does the same in production: about 19 % of its P picks
# fall in the first 0.2 s of the window. The spike decays inside the first
# second, and the S curve does not do it. Blank that second of P before peak
# picking. Playback and real time both build a fresh 60 s tensor, so both go
# through this path. The next window overlaps by timeShift (20 s), which is
# longer than the blank, so a real P in that second is still inside the
# previous window. EQCCT and EQTransformer are not in this set.
P_EDGE_SECONDS = 1.0
P_EDGE_ARCHITECTURES = frozenset(("phasenet", "utrans_eqcct"))


def _p_edge_samples(sampling_rate, n_samples):
    """Leading P samples that belong to the window edge, at this sample rate.

    The count is one second of the samples that entered the network.
    prepare_station_chunk resamples to 100 Hz when the trace is not already
    there, and the rate stored on the window is the rate after that step.
    A trace that stays at 40 Hz blanks 40 samples; one that stays at 250 Hz
    blanks 250. The duration does not depend on a fixed 100 Hz.
    """
    n_samples = int(n_samples or 0)
    try:
        rate = float(sampling_rate)
    except (TypeError, ValueError):
        rate = 0.0
    if not np.isfinite(rate) or rate <= 0.0:
        # Header missing. The tensors in this predictor are 60 s windows,
        # so the curve length still defines the rate.
        rate = (n_samples / 60.0) if n_samples else 0.0
    n_blank = int(round(P_EDGE_SECONDS * rate))
    if n_samples:
        n_blank = min(n_blank, n_samples)
    return max(0, n_blank)


# @ray.remote
def parallel_predict(predict_args, model=None, precision="bf16"):
    # stream, pos, st, args, out_dir= predict_args
    # stream, pos, st, args = predict_args
    stream, st, args = predict_args
    architecture = args.get("architecture", "eqcct")
    if model is None:
        try:
            loader = MODEL_LOADERS[architecture]
        except KeyError:
            raise ValueError(
                f"Unknown picker architecture {architecture!r}. "
                f"Valid options: {sorted(MODEL_LOADERS)}"
            )
        # Task mode loads a new model on every station. The policy has to be
        # set before that construction. Model-actor mode passes the model it
        # already loaded under the same policy.
        from .precision import begin_precision
        begin_precision(precision)
        model = loader(args["p_model"], args["s_model"])
    start_Predicting = time.time()
    meta, data_set = _readnparray(stream, args, st)
    batch_sizes = [args['batch_size'], 500, 250, 100, 50, 10, 5, 1]
    i = 0
    while True:
        # try:
        norm_mode = args['normalization_mode']
        # PhaseNet applies the official sliding mean/std inside its loader.
        if architecture == 'phasenet':
            norm_mode = 'none'
        params_pred = {'batch_size': batch_sizes[i], 'norm_mode': norm_mode}
        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
        predicted = model.predict(pred_generator, verbose=0)
        if architecture == "eqtransformer":
            predD, predP, predS = predicted
            predD = np.asarray(predD, dtype=np.float32)
            predP = np.asarray(predP, dtype=np.float32)
            predS = np.asarray(predS, dtype=np.float32)
            detection_memory = []
            picks = []
            for ix in range(len(predP)):
                window_picks, detection_memory = _eqt_window_picks(
                    meta, args, predD[ix], predP[ix], predS[ix], detection_memory, ix
                )
                picks.extend(window_picks)
            end_Predicting = time.time()
            delta = end_Predicting - start_Predicting
            info = f"[{datetime.now()}] {st}: Finished the prediction in {round(delta,2)}s."
            return {"picks": picks, "info": info}
        predP, predS = predicted
        predP = np.asarray(predP, dtype=np.float32)
        predS = np.asarray(predS, dtype=np.float32)
        if architecture in P_EDGE_ARCHITECTURES and predP.ndim >= 2 and predP.shape[1] > 0:
            n_blank = _p_edge_samples(meta.get("sampling_rate"), predP.shape[1])
            if n_blank > 0:
                predP = np.array(predP, copy=True)
                predP[:, :n_blank, ...] = 0.0

        #log.write(np.shape(predP))
        detection_memory = []
        prob_memory=[]
        picks = []
        for ix in range(len(predP)):
            Ppicks, Pprob =  _picker(args, predP[ix,:, 0])   
            Spicks, Sprob =  _picker(args, predS[ix,:, 0], 'S_threshold')
            #log.write(np.shape(Ppicks))

            #### Add option to activate write picks in csv file
            pick,detection_memory,prob_memory=_output_dict_prediction(meta, Ppicks, Pprob, Spicks, Sprob, detection_memory, prob_memory, ix,len(predP),len(predS))
            picks.append(pick)
                                        
        end_Predicting = time.time() 
        delta = (end_Predicting - start_Predicting) 
        
        info = f"[{datetime.now()}] {st}: Finished the prediction in {round(delta,2)}s."
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

def safe_trim_edge_pad(tr, starttime, endtime, nearest_sample=False):
    tr_cut = tr.copy()

    tr_cut.trim(
        starttime=starttime,
        endtime=endtime,
        pad=False,
        nearest_sample=nearest_sample
    )

    if tr_cut.stats.npts == 0:
        return None

    delta = tr_cut.stats.delta
    expected_npts = int(round((endtime - starttime) / delta)) + 1

    pad_left = int(round((tr_cut.stats.starttime - starttime) / delta))
    pad_left = max(0, pad_left)

    pad_right = expected_npts - tr_cut.stats.npts - pad_left
    pad_right = max(0, pad_right)

    data = np.asarray(tr_cut.data)

    if pad_left > 0 or pad_right > 0:
        data = np.pad(data, (pad_left, pad_right), mode="edge")

    tr_cut.data = data[:expected_npts]
    tr_cut.stats.starttime = starttime

    return tr_cut

def get_stream_filtered(net_sta: str,start,end,files):

    net, sta  = net_sta.split(".")
    st = obspy.Stream()
    for fn in files:
        try:
            temp_st = obspy.read(fn)
        except Exception as e:
            print(f"[READ ERROR] {fn}: {e!r}")
            continue

        try:
            # temp_st.merge(fill_value=0)
            temp_st.merge(method=1, fill_value="interpolate")
        except Exception as e:
            print(f"[MERGE ERROR] {fn}: {e!r}")
            try:
                # temp_st.merge(fill_value=0)
                temp_st.merge(method=1, fill_value="interpolate")
            except Exception as e2:
                print(f"[MERGE RETRY FAILED] {fn}: {e2!r}")

        # Select only the requested net/station before adding
        temp_st = temp_st.select(network=net, station=sta)
        if temp_st:
            st += temp_st

    st_trimmed = obspy.Stream()
    for tr in st:
        tr_new = safe_trim_edge_pad(tr, start, end)
        if tr_new is not None:
            st_trimmed.append(tr_new)
    st = st_trimmed


    if len(st) == 0:
        print(f"Empty stream for {net}.{sta} in window [{start} .. {end}]")
        return None

    print(f"Stream ready: {net}.{sta} in window [{start} .. {end}] with {len(st)} traces")
    return st

def prepare_station_chunk(st_or_files, net_sta, t0, t1, stations_filters=None,
                          default_band=(1.0, 45.0)):

    net, sta = net_sta.split(".")

    # -------- REAL-TIME --------
    if isinstance(st_or_files, obspy.Stream):
        st = st_or_files.select(network=net, station=sta).copy()
        if not st:
            return None

        # 1) Merge + detrend(linear) + demean
        # st.merge(fill_value=0)
        st.merge(method=1, fill_value="interpolate")
        st.detrend("linear")
        st.detrend("demean")

        # 2) Fixed 5-second taper (exact legacy behavior)
        try:
            max_percentage = 5.0 / (st[0].stats.delta * st[0].stats.npts)
            st.taper(max_percentage=max_percentage, type="cosine")
        except Exception:
            pass

        # 3) Bandpass per station (DF-like or list-of-dicts spec)
        fmin, fmax = default_band
        sta_name = st[0].stats.station
        if stations_filters is not None:
            # Try DataFrame-like
            try:
                fmin = float(stations_filters[stations_filters.sta == sta_name].iloc[0]["hp"])
                fmax = float(stations_filters[stations_filters.sta == sta_name].iloc[0]["lp"])
            except Exception:
                # Try list-of-dicts with "key" and "filter"
                try:
                    filt = next(d for d in stations_filters if d["key"].split(".", 1)[-1] == sta_name)["filter"]
                    fmin, fmax = sorted(float(x) for x in filt[filt.index("(")+1:filt.index(")")].split(",")[1:3])
                except Exception:
                    pass
        st.filter(type="bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)

        # 4) Resample to 100 Hz if required (interpolate first, fallback to resample)
        if any(tr.stats.sampling_rate != 100.0 for tr in st):
            try:
                st.interpolate(100.0, method="linear")
            except Exception:
                for tr in st:
                    tr.resample(100.0)
        
        global_start = min(tr.stats.starttime for tr in st)
        global_end = max(tr.stats.endtime for tr in st)
        new_st = obspy.Stream()
        for tr in st:
            tr_new = safe_trim_edge_pad(tr, global_start, global_end)
            if tr_new is not None:
                new_st.append(tr_new)
        st = new_st

        return st

    # -------- PLAYBACK: robust load/clean/filter + exact trim to [t0, t1] --------
    st = obspy.Stream()

    # Load and select only the target net.sta
    for fn in st_or_files:
        try:
            tmp = obspy.read(fn)
        except Exception:
            continue
        tmp = tmp.select(network=net, station=sta)
        if tmp:
            st += tmp

    if not st:
        return None

    # Clean masked arrays and non-finite values before merge/filter
    for tr in st:
        data = tr.data
        try:
            if np.ma.isMaskedArray(data):
                data = data.filled(0)
        except Exception:
            pass
        data = np.asarray(data)
        if not np.isfinite(data).all():
            bad = ~np.isfinite(data)
            if bad.any():
                data = data.copy()
                data[bad] = 0
        tr.data = data

    # Merge with fill_value=0 (bridge gaps with zeros)
    try:
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        st.merge(fill_value="interpolate")
    # try:
    #     st.merge(method=1, fill_value=0)  # method=1 keeps gaps simple; fill with zeros
    # except Exception:
    #     st.merge(fill_value=0)

    # Detrend + demean
    try:
        st.detrend("linear")
        st.detrend("demean")
    except Exception:
        pass

    # Fixed 5-second taper (stable across window lengths)
    try:
        max_percentage = 5.0 / (st[0].stats.delta * st[0].stats.npts)
        st.taper(max_percentage=max_percentage, type="cosine")
    except Exception:
        pass

    # Station-specific bandpass if available
    fmin, fmax = default_band
    if stations_filters is not None:
        # Try list-of-dicts first
        try:
            filt = next(d for d in stations_filters if d["key"].split(".", 1)[-1] == sta)["filter"]
            fmin, fmax = sorted(float(x) for x in filt[filt.index("(")+1:filt.index(")")].split(",")[1:3])
        except Exception:
            # Fallback to DataFrame-like
            try:
                fmin = float(stations_filters[stations_filters.sta == sta].iloc[0]["hp"])
                fmax = float(stations_filters[stations_filters.sta == sta].iloc[0]["lp"])
            except Exception:
                pass

    # Apply bandpass (fallback per-trace if needed)
    try:
        st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    except Exception:
        new_st = obspy.Stream()
        for tr in st:
            try:
                tr2 = tr.copy()
                tr2.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
                new_st += tr2
            except Exception:
                new_st += tr.copy()
        st = new_st

    # Resample to 100 Hz if required
    if any(abs(tr.stats.sampling_rate - 100.0) > 1e-6 for tr in st):
        try:
            st.interpolate(100.0, method="linear")
        except Exception:
            for tr in st:
                tr.resample(100.0)

    new_st = obspy.Stream()
    for tr in st:
        tr_new = safe_trim_edge_pad(tr, t0, t1)
        if tr_new is not None:
            new_st.append(tr_new)
    st = new_st

    # Sanitize to float32 and ensure finite values
    for tr in st:
        arr = np.asarray(tr.data)
        if not np.isfinite(arr).all():
            bad = ~np.isfinite(arr)
            if bad.any():
                arr = arr.copy()
                arr[bad] = 0
        tr.data = arr.astype(np.float32, copy=False)

    return st


def _readnparray(stream, args, st_name):
    # 1) Station-prepared Stream
    if args["playback"]:
        st = prepare_station_chunk(
            args["files"], args["station"], args["t_ini"], args["t_end"],
            stations_filters=args.get("stations_filters")
        )
    else:
        st = prepare_station_chunk(
            stream, args["station"], args["t_ini"], args["t_end"],
            stations_filters=args.get("stations_filters")
        )
    if not st:
        raise RuntimeError("Empty stream: no traces loaded.")

    span_start = min(tr.stats.starttime for tr in st)
    span_end   = max(tr.stats.endtime   for tr in st)

    if not args["playback"]:
        span_start = span_start + args['filterShift']

    # Map station components
    components = {tr.stats.channel[-1]: tr for tr in st}
    st_Z = st.select(channel="*Z")
    channel_out = st_Z[0].stats.channel if len(st_Z) > 0 else st[0].stats.channel

    rate_trace = st_Z[0] if len(st_Z) > 0 else st[0]
    meta = {
        "start_time": span_start,
        "end_time":   span_end,
        "trace_name": f"{st_name}.{st[0].stats.location}.{channel_out}",
        # Rate after prepare_station_chunk, which resamples anything that is
        # not already 100 Hz. The P-edge blank uses this value.
        "sampling_rate": float(rate_trace.stats.sampling_rate),
    }

    data_set = {}
    st_times = []

    # Preferred component mapping per column
    components_list = [
        ['E', '1'],  # Column 0
        ['N', '2'],  # Column 1
        ['Z']        # Column 2
    ]

    # 3) Window construction
    if args["playback"]:

        # 60 s windows with overlap given in seconds
        window_sec = 60.0
        overlap_sec = float(args.get("overlap", 0.0))

        if overlap_sec < 0:
            overlap_sec = 0.0

        if overlap_sec >= window_sec:
            raise ValueError(
                f"Invalid overlap={overlap_sec}. "
                f"Overlap must be smaller than the window length ({window_sec} s)."
            )

        step_sec = int(round(window_sec - overlap_sec))
        step_sec = max(1, step_sec)

        current = span_start

        while current < span_end:
            window_end = current + window_sec

            st_times.append(str(current).replace('T', ' ').replace('Z', ''))
            npz_data = np.zeros((6000, 3))

            for col_idx, comp_options in enumerate(components_list):
                for comp in comp_options:
                    if comp in components:
                        tr = components[comp].copy().slice(
                            current,
                            window_end,
                            nearest_sample=False
                        )
                        data = tr.data[:6000]
                        if len(data) < 6000:
                            data = np.pad(data, (0, 6000 - len(data)), 'constant')
                        npz_data[:, col_idx] = data
                        break

            key = str(current).replace('T', ' ').replace('Z', '')
            data_set[key] = npz_data

            current += step_sec

    else:
        # Real-time: single window.
        # If we have at least 60 s, prefer the last 60 s; otherwise use the full span.
        if (span_end - span_start) >= 60:
            win_start = span_end - 60
            win_end   = span_end
        else:
            win_start = span_start
            win_end   = span_end

        st_times.append(str(win_start).replace('T', ' ').replace('Z', ''))
        npz_data = np.zeros((6000, 3))
        for col_idx, comp_options in enumerate(components_list):
            for comp in comp_options:
                if comp in components:
                    tr = components[comp].copy().slice(win_start, win_end, nearest_sample=False)
                    data = tr.data[:6000]
                    if len(data) < 6000:
                        data = np.pad(data, (0, 6000 - len(data)), 'constant')
                    npz_data[:, col_idx] = data
                    break

        key = str(win_start).replace('T', ' ').replace('Z', '')
        data_set[key] = npz_data

    meta["trace_start_time"] = st_times

    # 4) Populate basic metadata
    try:
        meta.update({
            "receiver_code": st[0].stats.station,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })
    except Exception:
        meta.update({
            "receiver_code": st_name,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })

    return meta, data_set

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
        if mode in (None, 'none'):
            return data
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

def _eqt_matches(args, detection, pred_p, pred_s):
    """Associate P and S peaks with a detection, as in EqT mseed_predictor.

    A pick is kept only when the detection curve stays above
    detection_threshold (default 0.3, the published mseed default) for at
    least 0.1 s. P is the strongest peak in the second before the detection
    and at least 0.1 s before S. S is the earliest peak inside the detection.
    Peaks of P or S outside a detection are dropped.
    """
    from obspy.signal.trigger import trigger_onset

    det = np.asarray(detection, dtype=np.float64).reshape(-1)
    yh_p = np.asarray(pred_p, dtype=np.float64).reshape(-1)
    yh_s = np.asarray(pred_s, dtype=np.float64).reshape(-1)
    n = min(det.size, yh_p.size, yh_s.size)
    det, yh_p, yh_s = det[:n], yh_p[:n], yh_s[:n]
    det_thr = float(args.get("detection_threshold", 0.3))
    intervals = trigger_onset(det, det_thr, det_thr)
    p_peaks = _detect_peaks(yh_p, mph=args["P_threshold"], mpd=1)
    s_peaks = _detect_peaks(yh_s, mph=args["S_threshold"], mpd=1)
    matches = []
    for onset, offset in intervals:
        onset = int(onset)
        offset = int(offset)
        if offset - onset < 10:
            continue
        s_inside = [int(sample) for sample in s_peaks if onset < int(sample) < offset]
        s_index = s_inside[0] if s_inside else None
        if s_index is not None:
            p_inside = [int(sample) for sample in p_peaks if onset - 100 < int(sample) < s_index - 10]
        else:
            p_inside = [int(sample) for sample in p_peaks if onset - 100 < int(sample) < offset]
        p_index = max(p_inside, key=lambda sample: yh_p[sample]) if p_inside else None
        if p_index is None and s_index is None:
            continue
        matches.append({
            "onset": onset,
            "p_index": p_index,
            "p_prob": None if p_index is None else float(np.round(yh_p[p_index], 3)),
            "s_index": s_index,
            "s_prob": None if s_index is None else float(np.round(yh_s[s_index], 3)),
        })
    return matches


def _eqt_window_picks(meta, args, detection, pred_p, pred_s, detection_memory, idx):
    """One pick dict per detection. Overlapping windows skip a repeat within 2 s."""
    start_time = meta["trace_start_time"][idx]
    try:
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    picks = []
    for match in _eqt_matches(args, detection, pred_p, pred_s):
        ev_start = start_time + timedelta(seconds=match["onset"] / 100.0)
        if any(abs((ev_start - previous).total_seconds()) < 2 for previous in detection_memory):
            continue
        pick, _, _ = _output_dict_prediction(
            meta,
            [match["p_index"]],
            [match["p_prob"]],
            [match["s_index"]],
            [match["s_prob"]],
            [],
            [],
            idx,
            0,
            0,
        )
        picks.append(pick)
        detection_memory.append(ev_start)
    return picks, detection_memory


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
    except Exception as e:
        print(str(e))
        Ppickall.append(None)
        Pproball.append(None)
    
    return Ppickall, Pproball
