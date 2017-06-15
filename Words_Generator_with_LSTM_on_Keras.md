---
layout: default2
title: LSTM
permalink: /LSTM
---


# Words Generator with LSTM on Keras

##### Wei-Ying Wang 6/13/2017

This is a simple LSTM model built with Keras. The purpose of this tutorial is to help you gain solid understanding of LSTM model and the usage of Keras.

The code here wants to build [Karpathy's Character-Level Language Models](https://gist.github.com/karpathy/d4dee566867f8291f086) with Keras. Karpathy posted the idea on his [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). It is a very fun blog post, which generated shakespear's article, as well as Latex file with many math symbols. I guess we will never run out of papers this way...

I found a lot of "typo" in the official document of [keras](keras.io). Don't be too harsh to them; it is expected since keras is a huge module and it is hard for their document to keep on track of their own update. I write this tutorial to help people that want to try LSTM on Keras. I spent a lot of time looking into the script of keras, which can be found in your python folder:
```
\Lib\site-packages\keras
```

The following code is running on 
```
Python 3.6.0 (v3.6.0:41df79263a11, Dec 23 2016, 08:06:12) [MSC v.1900 64 bit (AMD64)]

keras version 1.2.2
```


```python
import numpy as np
from __future__ import print_function
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
import pickle
import AuxFcn
```

    Using TensorFlow backend.
    

## Data input

A tiny part of the code in this section here is using Karpathy's code in [here](https://gist.github.com/karpathy/d4dee566867f8291f086). 

The original shakespeare data has 65 distint characters. To relieve some computational burden, I reduced it into 36 characters with my own function `AuxFcn.data_reducing()`. Basically, I change all the uppercase letters to lowercase one, and only retain
```
",.?! \n:;-'"
```
characters. Should any other characters appear in the raw data, I simply change it into space character.

In the end we tranfer the strings of size `n` into a list of integers, `x`. You can convert the interger back to string by dictionary `ix2char`.


```python
#%%
data = open('tinyShakespeare.txt', 'r').read() 
data = AuxFcn.data_reducing(data)

chars = list(set(data))
n, d = len(data), len(chars)
print('data has %d characters, where %d of them are unique.' % (n, d))
char2ix = { ch:i for i,ch in enumerate(chars) }
ix2char = { i:ch for i,ch in enumerate(chars) }
#%% from text data to int32
x = [char2ix[data[i]] for i in range(len(data))]
```

    data has 1115394 characters, where 36 of them are unique.
    


```python
char2ix.keys()
```




    dict_keys(['v', 'k', '!', 's', 'n', 'e', 'c', '.', 'f', '-', 'r', 'w', 'q', 'z', "'", '\n', 'y', 'h', 'u', ';', ' ', 'o', 'a', 'd', 'i', '?', 'l', 'm', 'j', 'b', 't', ':', 'g', 'p', 'x', ','])




```python
pickle.dump(ix2char,open('dic_used_LSTM_16_128.pkl','wb')) 
# You will want to save it, since everytime you will get different ix2char dictionary, since you have use set() before it.
```

# First model: Using 16 words to predict the next word

Our model will only make prediction based on the previous `T` characters. This is done by setting the time_step, $T$ by `T=16.`

First we have to convert x into onehot representation. So we convert `x` (which is a interger list of size `(n,)`) to `x_nTd`. Also, we set the prediction `y_n`.

 * `x_ntd`: numpy `Boolean` array of size `(n,T,d)`, where `d` is the number of possible characters (`d=36`).
 * `y_n`: numpy `Boolean` array of size `(n,d)`. 
  - For i=1,2,...,n, `y_n[i,:]=x[i+1,0,:]`. 

Note that I only use `N=200000` samples to build the model.


```python
T=16
x_nTd,y_n = AuxFcn.create_catgorical_dataset(x, d,T)

N = 200000
x_tmp,y_tmp = x_nTd[:N,:,:],y_n[:N,:]
```


```python
print('This are 15 of thesamples of a slice of `x_tmp`:\n')
print(AuxFcn.translate(x_tmp[200:215,-1,:],ix2char))
print('\n The following is corresponding `y`, You can see that `y_n[i,:]=x[i+1,0,:]`:\n')
print(AuxFcn.translate(y_tmp[200:215,:],ix2char))
```

    This are 15 of thesamples of a slice of `x_tmp`:
    
    cius is chief e
    
     The following is corresponding `y`, You can see that `y_n[i,:]=x[i+1,0,:]`:
    
    ius is chief en
    

## Defining an LSTM layer 

 1. In the following, we will assign the first layer to be LSTM
    ```
    m=128
    model.add(LSTM(m, input_shape=(T, d))).
    ```
    This means: when unroll this recurrent layer, we will see:

      * 6 LSTM cells, that output T hidden units $(h_1,...,h_T)$, where each unit is a vector of size $m$. 
        - Note that there are also T state units $(s_1,...,s_T$, that only used between the LSTM cells in the same layer.
          - the state units (AKA recurrent units) controls long term information, which will be controlled by forget gate. 
      * The input layer are T units  $(x_1,...,x_T)$, each unit is a vector of size `d`
      * Note that every LSTM cell **shares** the same parameter.

 2. The next layer is the output layer, using `softmax`. Note that the softmax only applies on the information of $h_T$, the last activation of $h$. 

 3. The structure of the unrolled neural network is:
    ```
                          y
                          |
    h_1 -- h_2 -- ... -- h_T
     |      |     ...     |
    x_1    x_2    ...    x_T

    ```

### Parameters in LSTM layer

I will give a little explaination on the number of parameter of the LSTM layer.

The calculation of $h_t$, $t=1,2,...,T$, requires:$$U*h_{t-1}+W*x_t+b,$$ for        
 
 - $U = (U_f,U_g,U_o,U_i)$,
 - $W = (W_f,W_g,W_o,W_i)$, and
 - $b = (b_f,b_g,b_o,b_i)$, where
   - $f$: forget gate
   - $g$: external input gate 
   - $o$: output gate
   - $i$: input 
     
Note that each $U$ is (m,m), each $W$ is (m,d), each $h$ is (m,), we will totally need
$$4\cdot(m^2+m\cdot d+m)$$ parameters.

### Forward Propagation

The forward propagation will be: set $h_0=\bf 0$ and $s_0=\bf 0$, then
  
  1. input $x_1$, then calculate $h_1$ and $s_1$, then
  2. input $x_2$, then calculate $h_2$ and $s_2$, and so on
  3. Unitl obatain $h_T$
  



```python
m=128
model = Sequential()
model.add(LSTM(m, input_shape=(T, d)))
model.add(Dense(d,activation='softmax'))
#%%
adam = Adam(clipvalue=1)# any gradient will be clipped to the interval [-1,1]
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    lstm_2 (LSTM)                    (None, 128)           84480       lstm_input_2[0][0]               
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 36)            4644        lstm_2[0][0]                     
    ====================================================================================================
    Total params: 89,124
    Trainable params: 89,124
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    

## Training the model

 * Note that we have to set `batch_size` parameter when training. This is the size of samples used when calculate the stochastic gradient descent (SGD)

     * If input `x` has N samples (i.e. `x.shape[0]=N`) and `batch_size=k`, then for every epoch

         * `model.fit` will run $\frac{N}{k}$ iterations, and
         * each iteration calculate SGD with $k$ samples.
     * That is, in every epoch the training procedure will "sweep" all the samples. So, small 'batch_size' means the weight will be updated more times in a epoch.
      
 * You can estimate how many time it will take for 1 epochs. By setting
     ```
     initial_epoch=0
     nb_epoch=1
     ```
   And if you set `initial_epoch=1` in the next time you execute the `model.fit`, it will initialize the weights with your previous result. That is pretty handy.
   
 * Usually, to train RNN, you will want to turn off the `shuffle`. `shuffle=True` parameter will shuffle the sampels in each epoch (so that SGD will get random sample in different epoch). Since we are using $T$ time steps to build the model, it is no matter you turn on or not. However, if you somehow build a model with `batch_size=1` and `stateful=True`, you will need to turn of the shuffle. (See Appendix for the `stateful` argument)
 
 
 


```python
history = model.fit(x_tmp, y_tmp,
                  shuffle=False,
                  batch_size=32,
                  nb_epoch=10,
                  verbose=2, # verbose controls the infromation to be displayed. 0: no information displayed
                  initial_epoch=0)
#%%
AuxFcn.print_model_history(history)

```

 * Save and load model
    You can save model by
    ```
    model.save('keras_char_RNN')
    ```
    And access it with:
    ```
    model = keras.models.load_model('keras_char_RNN')
    ```
 * The training procedure will not give a good accuration, I got accuration about 0.6. But it is expected, since if you got 90% correction rate, then Shakespeare is just a word machine without any inspiration... i.e. The model learned is Shakespear's grammar, structures, and words. Far from the idea or spirit of Shakespear.

## Fun time: Generate the txt
To have fun fast, you can load the model I generated, don't forget to load the dictionary `ix2char`


```python
my_ix2char = pickle.load( open( "dic_used_LSTM_16_128.pkl", "rb" ) )
model_trained = load_model('keras_char_RNN')
```

The following code generates the text


```python
#%%
initial_x = x_nTd[250000,:,:]
words = AuxFcn.txt_gen(model_trained,initial_x,n=1000,diction=my_ix2char) # This will generate 100 words.
print(words)
```

    s
    to live the dees in, my lord, as what the wars to stam;
    those good nows to my your past,
    i will a cack up bodd withes s
    are read'd
    angefing to hame made men gave.
    
    queen margaret:
    he caners, and scorn'd the there, no worshieful geafles and to be changes onde:
    but they grant y be weeping
    the kinging in pain to ends you to rome,
    all tongage own business?
    
    first murderer:
    said, lady, him she's friends,
    wherein i wish here must?
    
    clarence:
    o, lady moretworanc which in any turbted; but we my young direffally commended of thems.
    thou launtient fambiness
    she my ligest with?
    but mess yours thy knews:
    now it shall i have i lo dows, ladies, when land
    thee are the lie;
    and have too thou are those
    than will reckects
    flatwedlen my wife come a lord the gone.
    
    second murderer:
    no blood dawger but itself: the plague is the kill!
    
    gloucester:
    say! marrying wry plebeing: ran
    up hour namelier in him.
    weretimy i shall fears you good offend atonest and fuelong,
    to destine and a nasters,
    well see what be 
    

## Appendix

### Confusion about `batch_input_shape`, `input_shape`, `batch_shape`

I check the keras code to derive the following statement.

 * First, when mentioning "batch", it always means the size of sample used in stochastic gradient descent (SGD).

    When build the first layer, if using `batch_input_shape` in `model.add(...)` and set the batch size to `10`, e.g.
    ```
    model.add(LSTM(m,batch_input_shape=(10,T,d)))
    ```
    Then when you doing `model.fit()`, you must assign `batch_size=10`, otherwise the program will give error message.

 * Consider this is a little bug, if you didn't assign `batch_size` in `model.fit()`, the SGD will run with default `batch_size=32`, which is not consistent with the `batch_input_shape[0]`. This is will raise `ValueError`

 * A better way is not to specify `batch_input_shape` when define the first layer; instead, using `input_shape=(T,d)`, which will equivalently assign 
    ```
    batch_input_shape=(None,T,d)
    ```
    And when you want to train the model, assign `batch_size` in `model.fit()`

    This way one can input any number of samples in the model to get predictions, otherwise, if you use `batch_input_shape` then the input must be consistent to the shape.

### What is `stateful` parameter
You might be wondered what is `stateful` argument when building the first LSTM layer. i.e.
```
model.add(LSTM(...,stateful=False))
```

If using `stateful=True`, when parameter update by SGD for 1 batch (here we set `batchsize=10`), say we have the activation $h_1^\star,...,h_T^\star$ and $s_1^\star,...,s_T^\star$. Then, in the next batch, the $h_0$ will be set as $h_T^\star$  and the $s_0$ will be set as $s_T^\star$. 
The previous procedure doesn't make a lot of sense. I just put it the way so you can understand. So, when will we use `stateful=True`? For example: when every time step you want to output a prediction (rather than output a prediction using 6 time steps, as we are doing here) We will, in the end, build that word generator that using previous word to generate the next word, at that time, we will turn this parameter on.

The defaut value is `stateful=False`.

### The dropout in LSTM 
To have dropout (note that the website of [keras](keras.io) uses keyword 'dropout', which cannot run in this version), use the following keywords when building LSTM layer (i.e. `model.add(LSTM(...,dropout_W=0.2,dropout_U=0.2))`. The describtion I found in keras module is:
 ```
 dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
 dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections. 
 ```


