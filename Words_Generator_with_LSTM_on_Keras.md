
# Words Generator with LSTM on Keras

##### Wei-Ying Wang 6/13/2017

This is a simple LSTM model built with Keras. The purpose of this tutorial is to help you gain some understanding of LSTM model and the usage of Keras. This post is generated from jupyter notebook. You can download the .ipynb file, along with the material used here, at [My Github](https://github.com/wayinone/Char_LSTM)

The code here wants to build [Karpathy's Character-Level Language Models](https://gist.github.com/karpathy/d4dee566867f8291f086) with Keras. Karpathy posted the idea on his [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). It is a very fun blog post, which generated shakespear's article, as well as Latex file with many math symbols. I guess we will never run out of papers this way... Most of all, this seems to be a great starting point to understand recurrant networks.

I found a lot of "typo" in the official document of [keras](keras.io). Don't be too harsh to them; it is expected since keras is a extemely complicated module and it is hard for their document to keep on track of their own update. I write this tutorial to help people that want to try LSTM on Keras. I spent a lot of time looking into the script of keras, which can be found in your python folder:
```
\Lib\site-packages\keras
```

The following code is running on 
```
Python 3.6.0 (v3.6.0:41df79263a11, Dec 23 2016, 08:06:12) [MSC v.1900 64 bit (AMD64)]

keras version 1.2.2
```

# 1. Shakespeare vs the Counterfeit
Let's take a peek at the masterpiece:

>Second Citizen:<br />
Consider you what services he has done for his country?<br />
<br />
First Citizen:<br />
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.<br />
<br />
Second Citizen:<br />
Nay, but speak not maliciously.<br />
<br />
First Citizen:<br />
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.<br />
<br />
Second Citizen:<br />
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.<br />
<br />
First Citizen:<br />
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

Ane the following is the counterfeit:
>tyrrin:<br />
in this follow'd her emeth tworthbour both!<br />
the great of roguess and crave-<br />
down to come they made presence not been me would?<br />
<br />
stanley:<br />
my rogrer to thy sorrow and, none.<br />
<br />
king richard iii:<br />
o, lading freeftialf
the brown'd of this well was a
manol, let me happy wife on the conqueser love.<br />
<br />
king richard iii:<br />
no, tyrend, and only his storces wish'd,
as there, and did her injury.<br />
<br />
hastings:<br />
o, shall you shall be thee,
the banters, that the orditalles in provarable-shidam; i did not be so frangerarr engley!
what is follow'd hastely be good in my son.<br />
<br />
king richard iii:<br />
or you good thought,
were they hatenings at temper his falls,
firsh to by do all,
and adsime.
if i her joy.


It is amazing how similar (structurewise) between the real work and the conterfeit. This tutorial will tell you step by step how this can be down with keras, along with some of my notes about the usage of keras.



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
    

# 2. Data input

A small part of the code in this section is using Karpathy's code in [here](https://gist.github.com/karpathy/d4dee566867f8291f086). 

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
print('Data has %d ASCII characters, where %d of them are unique.' % (n, d))
char2ix = { ch:i for i,ch in enumerate(chars) }
ix2char = { i:ch for i,ch in enumerate(chars) }
#%% from text data to int32
x = [char2ix[data[i]] for i in range(len(data))]
```

    Data has 1115394 ASCII characters, where 36 of them are unique.
    


```python
char2ix.keys()
```




    dict_keys(['v', 'k', '!', 's', 'n', 'e', 'c', '.', 'f', '-', 'r', 'w', 'q', 'z', "'", '\n', 'y', 'h', 'u', ';', ' ', 'o', 'a', 'd', 'i', '?', 'l', 'm', 'j', 'b', 't', ':', 'g', 'p', 'x', ','])




```python
pickle.dump(ix2char,open('dic_used_LSTM_16_128.pkl','wb')) 
# You will want to save it, since everytime you will get different ix2char dictionary, since you have use set() before it.
```

# 3. The model: Using 16 words to predict the next word

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
print('These are 15 of the samples of a slice of `x_tmp`:\n')
print(AuxFcn.translate(x_tmp[200:215,-1,:],ix2char))
print('\nThe following is corresponding `y`, You can see that `y_n[i,:]=x[i+1,0,:]`:\n')
print(AuxFcn.translate(y_tmp[200:215,:],ix2char))
```

    These are 15 of the samples of a slice of `x_tmp`:
    
    hief enemy to t
    
    The following is corresponding `y`, You can see that `y_n[i,:]=x[i+1,0,:]`:
    
    ief enemy to th
    

## 3.1 Constructing an LSTM layer 

 1. In the following, we will assign the first layer to be LSTM
    ```
    m=128
    model.add(LSTM(m, input_shape=(T, d))).
    ```
    This means: when unroll this recurrent layer, we will see:

      * 16 LSTM cells (Since `T=16`), where the cell output T hidden units $(h_1,...,h_T)$, where each unit is a vector of size `m`. 
        - Note that there are also T state units $(c_1,...,c_T)$, that only used between the LSTM cells in the same layer.
          - the state units (AKA recurrent units) controls long term information, which will be controlled by forget gate. 
      * The input layer are T units  $(x_1,...,x_T)$, each unit is a vector of size `d`, the number of distinct characters.
      * Note that every LSTM cell **shares** the same parameter.

 2. The next layer is the output layer, using `softmax`. Note that the softmax only applies on the information of $h_T$, the last activation of $h$. 

 3. The structure of the unrolled neural network is (Also, take a look at Appendix 4, where a different architechure is defined):
    ```
                          y
                          |
    h_1 -- h_2 -- ... -- h_T
     |      |     ...     |
    x_1    x_2    ...    x_T

    ```

### Parameters in LSTM layer

I will give a little explaination on the numbers of parameter of a LSTM layer. For more detail information, you can visit [here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

The $t^{th}$ LSTM cell will produce two ouputs: the **output state** $h_t$, and the **cell state** $c_t$, both of them are `(m,)` vectors.

The calculation of the outputs of an LSTM cell, $h_t$, $t=1,2,...,T$, requires the computation of current input units after 4 gates: **forget gate, cell state gate, output gate, input gate**
$$U\cdot h_{t-1}+W\cdot x_t+b.$$ 
In particular:
$$f_t         = \sigma(U_f\cdot h_{t-1}+W_f \cdot x_{t-1}+b_f)$$
$$\bar{c}_t   = tanh(U_c\cdot h_{t-1}+W_c \cdot x_{t-1}+b_c)$$
$$o_t         = \sigma(U_o\cdot h_{t-1}+W_o \cdot x_{t-1}+b_o)$$
$$i_t         = \sigma(U_i\cdot h_{t-1}+W_i \cdot x_{t-1}+b_i)$$
  
where $\sigma$ is the activation function, and       
 
 - $U = (U_f,U_c,U_o,U_i)$,
 - $W = (W_f,W_c,W_o,W_i)$, and
 - $b = (b_f,b_c,b_o,b_i)$, where
   - $f$: index about **forget gate**
   - $\bar{c}$: index about **cell state gate**
   - $o$: index about **output gate**
   - $i$: index about **input gate**
     
Note that each $U_f,U_c,U_o,U_i$ is `(m,m)`, each $W$ is `(m,d)`, each $b$ is `(m,)`. Thus, in total we have
$$4\cdot(m^2+m\cdot d+m)$$ parameters.

The calculation of the output state $h_t$, and the cell state $c_t$ is as follows:
$$ c_t = f_t\cdot c_{t-1} +i_t \cdot \bar{c}_t$$
$$ h_t = o_t\cdot tanh(c_t)$$

As you can see,

 1. Each gate is only applied on the current input.
 2. The cell state unit is controlled by **forget gate, previous cell state unit** and **input gate**.
 3. The output unit is controlled by **output gate** and **cell state gate**.

### Forward Propagation

The forward propagation will be: set $h_0=\bf 0$ and $c_0=\bf 0$, then
  
  1. input $x_1$, then calculate $h_1$ and $c_1$, then
  2. input $x_2$, then calculate $h_2$ and $c_2$, and so on
  3. Unitl obatain $h_T$
  



```python
m=128
model = Sequential()
model.add(LSTM(m, input_shape=(T, d)))
model.add(Dense(d,activation='softmax'))
#%%
adam = Adam(clipvalue=1)# all the gradient will be clipped to the interval [-1,1]
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
    dense_2 (Dense)                  (None, 36)            4644        lstm_2[0][0]                     
    ====================================================================================================
    Total params: 89,124
    Trainable params: 89,124
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    

Note that the `Output Shape` are `(None, 128)` and `(None, 36)`, this means the model can receive dynamic batch size, i.e. if you input a batch of samples of size `k` (for calculate of SGD), the first layer will generate output of size `(k,128)`. Take a look at Appendix 1, where I explain `batch_input_shape`, `input_shape`, `batch_shape`.

## 3.2 Training the model

 * Note that we have to set `batch_size` parameter when training. This is the size of samples used when calculate the stochastic gradient descent (SGD)

     * If input `x` has N samples (i.e. `x.shape[0]=N`) and `batch_size=k`, then for every epoch

         * `model.fit` will run $\frac{N}{k}$ iterations, and
         * each iteration calculate SGD with $k$ samples.
     * That is, in every epoch the training procedure will "sweep" all the samples. So, small 'batch_size' means the weight will be updated more times in a epoch.
      
 * You can estimate how many time it will take for 1 epochs. By setting
     ```
     initial_epoch = 0
     nb_epoch = 1
     ```
   And if you set `initial_epoch=1` in the next time you execute the `model.fit`, it will initialize the weights with your previous result. That is pretty handy.
   
 * Usually, to train RNN, you will want to turn off the `shuffle`. `shuffle=True` parameter will shuffle the sampels in each epoch (so that SGD will get random sample in different epoch). Since we are using $T$ time steps to build the model, it is no matter you turn on or not. However, if you somehow build a model with `batch_size=1` and `stateful=True`, you will need to turn of the shuffle. (See Appendix for the `stateful` argument)
 
 
 


```python
history = model.fit(x_tmp, y_tmp,
                  shuffle=False,
                  batch_size=2^5,
                  nb_epoch=1, # adjust this to calculate more epochs.
                  verbose=0, # 0: no info displayed; 1: most info displayed;2: display info each epoch
                  initial_epoch=0)
#%%
# AuxFcn.print_model_history(history)

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
 * The training procedure will not give a good accuration, I got accuration about 0.63. But it is expected, if you got 90% correction rate, then Shakespeare is just a word machine without any inspiration... i.e. The model learned is Shakespear's grammar, structures, and words. Far from the idea or spirit of Shakespear.

# 4. Fun time: Generate the txt
To have fun fast, you can load the model I generated, which has ran about 60 epoch (each epoch took about 140s), don't forget to load the dictionary `ix2char` as well.


```python
my_ix2char = pickle.load( open( "dic_used_LSTM_16_128.pkl", "rb" ) )
model_trained = load_model('keras_char_RNN')
```

The following code generates the text


```python
#%%
initial_x = x_nTd[250000,:,:]
words = AuxFcn.txt_gen(model_trained,initial_x,n=1000,diction=my_ix2char) # This will generate 1000 words.
print(words)
```

    diens if praise his grapie,
    and now that were shake shame thee lawn. it make my his eyes?
    
    menenius:
    it was and hear me of mind.
    or with the death motions,
    mutines unhoo jeily her entertablaness in the queen the duke of you make by edward
    that office corriag
    withal feo!
    will it the fat; our poss, myshling withaby gop with smavis,
    i am, but all stands not.
    
    lady atus:
    was with the friends him, triud!
    
    hasting:
    well, yet threng forth that not a pail;
    thou deserve terry keeps, to know humbance it, and that they mugabless cabiling given
    burght with wile wondelvy!
    lord, sut cursied the gray to tell me, sites by dangerand great business;
    go fan
    a power to dies 't
    bul the volsciagfel'd,
    when have did is frame?
    
    cominius:
    behay, i will know the truft, we prome, but it intworty knee, our enemies,
    whose as him 'fiselfuld me that know no more;
    must not smead in shed reasons!
    
    gloucester:
    say, making high even: for day i thank you aid; be not.
    
    first murderer:
    so thou wife from mine,
    less very the
    

## Appendix

### 1. Confusion about `batch_input_shape`, `input_shape`, `batch_shape`

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

### 2. What is `stateful` parameter
You might be wondered what is `stateful` argument when building the first LSTM layer. i.e.
```
model.add(LSTM(...,stateful=False))
```

If using `stateful=True`, when parameter update by SGD for 1 batch (here we set `batchsize=10`), say we have the activation $h_1^\star,...,h_T^\star$ and $s_1^\star,...,s_T^\star$. Then, in the next batch, the $h_0$ will be set as $h_T^\star$  and the $s_0$ will be set as $s_T^\star$. 
The previous procedure doesn't make a lot of sense. I just put it the way so you can understand. So, when will we use `stateful=True`? For example: when every time step you want to output a prediction (rather than output a prediction using 6 time steps, as we are doing here) We will, in the end, build that word generator that using previous word to generate the next word, at that time, we will turn this parameter on.

The defaut value is `stateful=False`.

### 3. The dropout in LSTM 
To have dropout (note that the website of [keras](keras.io) uses keyword 'dropout', which cannot run in this version), use the following keywords when building LSTM layer (i.e. `model.add(LSTM(...,dropout_W=0.2,dropout_U=0.2))`. The describtion I found in keras module is:
 ```
 dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
 dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections. 
 ```


### 4. What is `return_sequences` parameter
This parameter is defined when assigning LSTM layer, e.g. 
```
LSTM(m, input_shape=(T, d), return_sequences=True)
```
This will ouput hidden units of each time, i.e. $h_1,h_2,...,h_T$ to output. By default it is set to `False` means the layer will only ouput $h_T$, the last time step.

Take a look at `Ouput Shape` at model summary:


```python
m=128
model = Sequential()
model.add(LSTM(m, input_shape=(T, d), 
          return_sequences=True))
model.add(Dense(d,activation='softmax'))
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    lstm_8 (LSTM)                    (None, 25, 128)       84480       lstm_input_7[0][0]               
    ____________________________________________________________________________________________________
    dense_7 (Dense)                  (None, 25, 36)        4644        lstm_8[0][0]                     
    ====================================================================================================
    Total params: 89,124
    Trainable params: 89,124
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    

 * Note that in this architecture, the `Output Shape` of the first layer is `(None,T,m)`, where `m` is the dimension of each $h_i$, $i=1,2,...,T$ .

     * Compare to the model we had, the first layer's `Output Shape` is `(None,m)`.

 * So, in this case, the architecture is:
    ```
    y_1    y_2           y_T
     |      |             |
    h_1 -- h_2 -- ... -- h_T
     |      |     ...     |
    x_1    x_2    ...    x_T
    ```
 * It is also clear that if you want to stack the LSTM models, you will have to on `return_sequences`.
