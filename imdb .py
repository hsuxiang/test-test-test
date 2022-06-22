#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_metrics as km
from sklearn.model_selection import train_test_split
from keras import regularizers


# In[2]:


data = keras.datasets.imdb


# In[3]:


max_word = 10000    #只考虑前1万个单词的编码，后面的都抛弃掉，所以编码的整数索引不会超过10000
(x_train, y_train), (x_test, y_test) = data.load_data(num_words = max_word)


# In[4]:


x_train.shape, y_train.shape  #y_train就是label


# In[5]:


x_test.shape, y_test.shape


# In[6]:


y_train


# In[7]:


word_index = data.get_word_index()  #下载word跟index对应的json文件


# word2vec：把文本训练成密集向量

# In[8]:


[len(x) for x in x_train]  #可以看到评论的长度不尽相同，因为要放到lstm/gru/全连接就需要固定长度，所以要填充


# In[9]:


x_train = keras.preprocessing.sequence.pad_sequences(x_train,300)  #短的就填充，长的就截断
x_test = keras.preprocessing.sequence.pad_sequences(x_test,300)


# In[10]:


[len(x) for x in x_train]


# In[11]:


x_train[0]   #可以看到每条评论已经被处理成长度为300的向量了


# In[12]:


x_train.shape  #可得训练数据大小为25000个，长度为300的


# In[13]:


x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size = 0.5, random_state = 0)
print(len(x_train),len(x_valid),len(x_test))


# In[14]:


model = keras.models.Sequential()
model.add(layers.Embedding(10000,50,input_length=300))  #(输入数据的维度，50，input_dim=输入的长度)
model.add(layers.Flatten())  #因为现在是三维的，[25000,300,5],flatten把后面两维展平
#model.add(layers.GlobalAveragePooling1D())
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.SimpleRNN(32))
#model.add(tf.keras.layers.GRU(32))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001),#在权重参数w添加L1正则化
                bias_regularizer=regularizers.l2(0.001),#在偏置向量b添加L2正则化
                activity_regularizer=regularizers.l1_l2(0.001)))#在输出部分添加L1和L2结合的正则化
#model.add(tf.keras.layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))  #二分类问题，最后输出是是或否，用sigmoid激活


# In[15]:


model.summary()


# In[16]:


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc', km.binary_precision(), km.binary_recall(),km.f1_score()]
              )   #二分类问题，loss用二元交叉熵


# In[17]:


from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
callbacks_list = [
    EarlyStopping(
        monitor = 'val_acc', #监控验证精度
        patience = 2, #如果验证精度多于三轮不改善则中断训练
        mode='max'),
    #在训练的过程中不断得保存最优的模型
    ModelCheckpoint(
        filepath = 'my_model_lstm.h5', #模型保存路径
        monitor = 'val_acc', #监控验证精度
        
        save_best_only = True, #如果val_accuracy没有改善则不需要覆盖模型
    )
]


# In[18]:


history = model.fit(x_train,y_train,epochs=15,batch_size=256,callbacks=callbacks_list,validation_data=(x_valid,y_valid))


# In[19]:


history.history.keys()


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(history.epoch, history.history.get('loss'), 'r', label='train_loss')
plt.plot(history.epoch, history.history.get('val_loss'),'b--', label='val_loss')
plt.legend()


# In[21]:


plt.plot(history.epoch, history.history.get('acc'), 'r', label='train_acc')
plt.plot(history.epoch, history.history.get('val_acc'), 'b--', label='val_acc')
plt.legend()  


# In[22]:


from tensorflow.keras.models import load_model
model = load_model('my_model_lstm.h5',custom_objects={'binary_precision':km.precision(),'binary_recall':km.recall(),'binary_f1_score':km.f1_score()})
model.evaluate(x_test, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




