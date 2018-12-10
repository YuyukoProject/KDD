
# coding: utf-8

# In[1]:


import pandas as pd,numpy as np
from keras.utils.np_utils import  to_categorical
from keras.layers import  Input,Dense,Activation,Dropout
from keras import  Sequential,optimizers,initializers
from sklearn.metrics import  accuracy_score,roc_curve,classification_report,confusion_matrix,auc
from xgboost import XGBClassifier 
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,minmax_scale
import time


# In[2]:


def data_std(x,min_max=False):
    # 将数据进行标准化
    if min_max:
        std=StandardScaler()
        x=std.fit_transform(x)
    else:
        x=minmax_scale(x)
    return x


# In[3]:


def data_propr(x,name=False):
    # 将对象型数据转为哑变量
    if name:
        x=pd.get_dummies(x,prefix=name,prefix_sep='_', drop_first=True)
    else:
        x=to_categorical(x)
    return x 


# In[4]:


def poc_plt(y_true,y_pred):
    '''根据数据结果绘制roc曲线
    输入的数据需要是以哑变量形式的'''
    y_te=np.array(y_true)
    y_pred=np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_te.ravel(), y_pred.ravel())
    #auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, c = 'r', lw = 2)
    plt.show()


# In[5]:


def pre2clf(prb):
    #根据所得概率将其转换为所需的分类结果
    y_pre_clf=[list(pre).index(max(pre)) for pre in prb]
    return y_pre_clf


# In[26]:


def neural_network(x_tr,y_tr,x_te,y_te,dum=False,min_max=False):
    start = time.clock()
    if dum:
        x_tr=data_std(x_tr,min_max=False)
        x_te=data_std(x_te,min_max=False)
    y_tr_dm=data_propr(y_tr,name=False)
    y_te_dm=data_propr(y_te,name=False)
    init = initializers.glorot_uniform(seed=1)
    simple_adam = optimizers.Adam()
    model = Sequential()
    model.add(Dense(units=5, input_dim=x_te.shape[1],kernel_initializer=init, activation='relu'))
    model.add(Dropout(1))
    model.add(Dense(units=6, kernel_initializer=init, activation='sigmoid'))
    model.add(Dropout(1))
    model.add(Dense(units=5, kernel_initializer=init, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
    model.fit(x_tr,y_tr_dm,verbose=0,class_weight={0:0.05,1:0.05,2:0.49,3:0.499,4:0.01})
    #返回模型的基础结构
    NN_class_repot=classification_report(model.predict_classes(x_te),y_te)
    NN_class_con=confusion_matrix(model.predict_classes(x_te),y_te)
    NN_class_pred=model.predict_classes(x_te)
    NN_class_pred_prob=model.predict_proba(x_te)
    #输出精确度
    print('神经网络耗时:',end='--')
    print(model.evaluate(x_te,y_te_dm))
    #简单绘制模型的roc曲线
    poc_plt(y_te_dm,NN_class_pred_prob)
    end = time.clock()
    #计算模型耗时
    print('神经网络耗时:%f'%(end-start))
    return NN_class_repot,NN_class_con,NN_class_pred,NN_class_pred_prob,model


# In[27]:


def xgboost_clf(x_tr,y_tr,x_te,y_te):
    start = time.clock()
    bst =XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, 
                   silent=True, objective='binary:logistic') 
    bst.fit(x_tr,y_tr)
    #返回模型基本结构
    XG_class_repot=classification_report(bst.predict(x_te),y_te)
    XG_class_con=confusion_matrix(bst.predict(x_te),y_te)
    XG_class_pred=bst.predict(x_te)
    XG_class_pred_prob=bst.predict_proba(x_te)
    #输出精确度
    print('xgboost精确度:',end='--')
    print(accuracy_score(y_te,bst.predict(x_te)))
    #简单绘制模型的roc曲线
    y_te_dm=data_propr(y_te,name=False)
    poc_plt(y_te_dm,XG_class_pred_prob)
    end = time.clock()
    print('xgboost耗时:%f'%(end-start))
    return XG_class_repot,XG_class_con,XG_class_pred,XG_class_pred_prob,bst


# In[28]:


def lightGBM_clf(x_tr,y_tr,x_te,y_te):
    start = time.clock()
    params = {
    'objective':'multiclass',
    'learning_rate':0.05,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':4,
    'num_class':5
    }
    lgb_train = lgb.Dataset(x_tr, label=y_tr,free_raw_data=False)
    lgb_eval=lgb.Dataset(x_te, label=y_te,free_raw_data=False)
    GBM = lgb.train(params,lgb_train,valid_sets=lgb_eval,verbose_eval=False)
    y_pre= GBM.predict(x_te, num_iteration=GBM.best_iteration)
    y_pred=pre2clf(y_pre)
    #返回模型基本结构
    lGBM_class_repot=classification_report(y_te,y_pred)
    lGBM_class_con=confusion_matrix(y_te,y_pred)
    lGBM_class_pred=y_pred
    lGBM_class_pred_prob=y_pre
    #输出精确度
    print('lightGBM精确度:',end='--')
    print(accuracy_score(y_te,y_pred))
    #绘制简单的roc曲线
    y_te_dm=data_propr(y_te,name=False)
    poc_plt(y_te_dm,lGBM_class_pred_prob)
    end = time.clock()
    print('lightGBM耗时:%f'%(end-start))
    return lGBM_class_repot,lGBM_class_con,lGBM_class_pred,lGBM_class_pred_prob,GBM


# In[38]:


def get_new_pre(pre_1,pre_2,pre_3,vote=False):
    # 根据前三个模型所得拟合概率,按要求返回最终的预测结果
    #计算分类概率,为绘制roc曲线做准备
    y_tol=np.concatenate((pre_1,pre_2,pre_3),axis=1)
    y_pre_pro=np.zeros(pre_1.shape)
    for i in range(len(y_tol)):
        a = np.argmax(y_tol[i])
        y_pre_i=mod2clf(a,pre_1[i],pre_2[i],pre_3[i])
        y_pre_pro[i]=y_pre_i    
    if vote:#根据投票返回分类结果
        y_te_1=pre2clf(pre_1)
        y_te_2=pre2clf(pre_2)
        y_te_3=pre2clf(pre_3)
        y_tol=np.concatenate((y_te_1,y_te_2,y_te_3),axis=1)
        for i in y_tol:# 计算分类结果
            y_pre[i]=[np.argmax(np.bincount(i))]
    else :#根据预测概率返回分类结果
        y_pre=pre2clf(y_pre_pro)
    return y_pre,y_pre_pro


# In[39]:


def mod2clf(i,pre_1,pre_2,pre_3):  
        y_pre_i=[]
        PIC=i//5
        if PIC<1:
            y_pre_i=pre_1
        elif PIC<2:
            y_pre_i=pre_2
        else:
            y_pre_i=pre_3
        return y_pre_i


# In[35]:


def get_pre(x_tr,y_tr,x_te,y_te,dum=False,min_max=False,vote=False):
    start = time.clock()
    y_te_p= data_propr(y_te,name=False)
    #神经网络处理
    NN_class_repot,NN_class_con,NN_class_pred,NN_class_pred_prob,model=neural_network(x_tr,y_tr,x_te,y_te)
    # XGnoost处理
    XG_class_repot,XG_class_con,XG_class_pred,XG_class_pred_prob,bst=xgboost_clf(x_tr,y_tr,x_te,y_te)
    #lightGBM处理
    lGBM_class_repot,lGBM_class_con,lGBM_class_pred,lGBM_class_pred_prob,Lgbm=lightGBM_clf(x_tr,y_tr,x_te,y_te)
    # 根据运行的结果重新进行分类
    to_pred_class,to_pred_prob =get_new_pre(NN_class_pred_prob,XG_class_pred_prob,lGBM_class_pred_prob,vote)
    #绘制三个模型汇总的结果
    print('综合模型精确度:',end='--')
    print(accuracy_score(y_te,to_pred_class))
    y_te_dm=data_propr(y_te,name=False)
    poc_plt(y_te_dm,to_pred_prob)
    # 分类汇报与混淆矩阵集合
    cla_rep={'神经网络':NN_class_repot,'XGboost':XG_class_repot,'Lgbm':lGBM_class_repot}
    con_met={'神经网络':NN_class_con,'XGboost':XG_class_con,'Lgbm':lGBM_class_con}
    #总体汇报
    total_rep={'cla_rep':cla_rep,'con_met':con_met}
    end = time.clock()
    print('模型耗时:%f'%(end-start))
    return to_pred_class,to_pred_prob,total_rep,model,bst,Lgbm


# In[12]:


# 新数据的判断
def model_predict(test,model1,model2,model3,vote=False):
    start = time.clock()
    model1_pre=model.predict_classes(test)
    model2_pre=bst.predict(x_te)
    model1_pre_pro=model1.predict_proba(test)
    model2_pre_pro=bst.predict_proba(x_te)
    model3_pre_pro=lGBM.predict(test, num_iteration=gbm.best_iteration)
    model3_pre=pre2clf(model3_pre_pro)
    to_pred_class,to_pred_prob =to_get_new_pre(model1_pre_pro,model2_pre_pro,model3_pre_pro,vote)
    end = time.clock()
    print('模型耗时:%f'%(end-start))
    return to_pred_class,to_pred_prob


# In[13]:


x_tr=pd.read_csv(r'C:\Users\acer\Desktop\data\gra\last\x_tr.csv',encoding='utf-8',delimiter=',')
y_tr=pd.read_csv(r'C:\Users\acer\Desktop\data\gra\last\y_tr.csv',encoding='utf-8',delimiter=',')

x_te=pd.read_csv(r'C:\Users\acer\Desktop\data\gra\last\x_te.csv',encoding='utf-8',delimiter=',')
y_te=pd.read_csv(r'C:\Users\acer\Desktop\data\gra\last\y_te.csv',encoding='utf-8',delimiter=',')


# In[40]:


to_pred_class,to_pred_prob,total_rep,model,bst,Lgbm=get_pre(x_tr,y_tr,x_te,y_te,dum=False,min_max=False,vote=False)


# In[15]:


#找出分歧预测,用于之后的深度挖掘或人工分析
def spe_ext(cla_pre,cla_pre_pro):
    pre_cla_pre=pre2clf(cla_pre_pro)
    cla_sub=cla_pre-pre_cla_pre
    index=np.where(cla_sub!=0)
    return index

