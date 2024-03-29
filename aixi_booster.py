import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import logging
from sklearn import linear_model
import pickle
import copy

from multiprocessing import Pool
import pickle
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import itertools
from sklearn import ensemble
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings("ignore")



def find_subsequence(seq, subseq):
    target = np.dot(subseq, subseq)
    candidates = np.where(np.correlate(seq,
                                       subseq, mode='valid') == target)[0]
    # some of the candidates entries may be false positives, double check
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask]

class pattern_lib(object):
    def __init__(self):
        try:
            with open('./pattern_lib.pkl', 'rb') as f:
                d= pickle.load(f)
                self.pattern_trial_count=d['pattern_trial_count']
                self.full_commands_count=d['full_commands_count']
                self.standard_commands_count=d['standard_commands_count']
                self.pattern_success_count=d['pattern_success_count']
                self.pattern_list=d['pattern_list']
        except Exception:
            #57 - текущее число команд.0я неинформативна
            self.standard_commands_count=57
            self.full_commands_count=self.standard_commands_count
            self.pattern_success_count=np.zeros(self.standard_commands_count+1)
            self.pattern_trial_count=np.zeros(self.standard_commands_count+1)
            self.pattern_list = list(range(self.full_commands_count+1))
        self.purge()
    def compilation(self,code):
        code=np.array(code)
        code=code[(code>0) & (code<=self.full_commands_count)]
        if len(code)<1:
            code=[0]
        while len(np.where(np.array(code)>self.standard_commands_count)[0])>0:
            idx=np.where(np.array(code)>self.standard_commands_count)[0][0]
            try:
                code_to_input=self.pattern_list[int(code[idx])]         
            except Exception:
                print('ERROR self.pattern_list[int(code[idx])]')
                print(idx)
                print(code[idx])
                print(len(self.pattern_list))
                print(self.pattern_list)
            if type(code_to_input)==type(1):
                code_to_input=[code_to_input]
            if idx<len(code):
                code = list(code[:idx])+list(code_to_input)+list(code[idx+1:])
            elif idx==0:
                code = list(code_to_input)+list(code[idx+1:])
            else:
                code = list(code[:idx])+list(code_to_input)
                
        return code
               
    def update(self,genom_list,profit_arr,trial_count=120):
        #перевести геномы в свёрнутую форму
        for i in range(len(genom_list)):
            genom_list[i]=np.round(genom_list[i][genom_list[i]>0.5])
        #чуваки, на которых надо учиться
        profit_bool=(profit_arr<=np.percentile(profit_arr,0.1))&(profit_arr<1)
        profit_index=np.where(profit_bool)[0]
        #оценить эффективность одиночных команд
        for i in range(len(genom_list)):
            genom = np.array(genom_list[i])
            success = np.isin(i,profit_index)
            for gen in range(len(self.pattern_list)):
                count = np.sum(genom==gen)
                self.pattern_trial_count[gen]+=count
                if success:
                    self.pattern_success_count[gen]+=count            
        if len(profit_index)>0:
            #взять рандомные выборки, поискать их в выигрышных и проигрышных генах
            for i in range(trial_count):
                genom_id=np.random.choice(profit_index)
                code_len=np.random.choice([2,2,2,3,3,4,5])
                genom = genom_list[genom_id]
                start_point=int(np.random.rand()*len(genom)-code_len)
                code_frag=genom[start_point:start_point+code_len]
                code_frag=code_frag[code_frag!=0]
                if len(code_frag)<=1:
                    continue
                #возможно, включает в себя функции
                #а это уже без функций
                code_frag_compiled=self.compilation(code_frag)
                if len(code_frag_compiled)<=1:
                    continue
                #найти в нашей базе фрагментов или создать новый
                frag_num=-1
                for j in range(1,len(self.pattern_list)):
                    if np.array_equal(code_frag_compiled,self.pattern_list[j]):
                        frag_num=j
                if frag_num==-1:
                    frag_num=len(self.pattern_list)
                    self.pattern_list.append(list(code_frag_compiled))
                    self.pattern_success_count=np.append(self.pattern_success_count,0)
                    self.pattern_trial_count=np.append(self.pattern_trial_count,0)
                    self.full_commands_count+=1
                #найти такие же во всех геномах
                for k in range(len(genom_list)):
                    genom_inner = genom_list[k]
                    if len(genom_inner)<len(code_frag):
                        continue
                    #найти здесь
                    fnd_count=len(find_subsequence(genom_inner,code_frag))#проверь, что на выходе правда не кортеж
                    #записать в статистику
                    self.pattern_trial_count[frag_num]+=fnd_count
                    success = np.isin(k,profit_index)
                    self.pattern_success_count[frag_num]+=fnd_count*success
                    #найти здесь compiled
                    fnd_count=len(find_subsequence(genom_inner,code_frag_compiled))
                    #записать в статистику
                    self.pattern_trial_count[frag_num]+=fnd_count
                    success = np.isin(k,profit_index)
                    self.pattern_success_count[frag_num]+=fnd_count*success
    def to_file(self):
        d={}
        d['pattern_trial_count']=self.pattern_trial_count
        d['full_commands_count']=self.full_commands_count
        d['standard_commands_count']=self.standard_commands_count
        d['pattern_success_count']=self.pattern_success_count
        d['pattern_list']=self.pattern_list
        with open('./pattern_lib.pkl', 'wb') as f:
                pickle.dump(d,f)
    def purge(self):
        full_commands_count=self.standard_commands_count
        pattern_list = self.pattern_list[:full_commands_count+1]
        pattern_trial_count = self.pattern_trial_count[:full_commands_count+1]
        pattern_success_count = self.pattern_success_count[:full_commands_count+1]
        for i in range(self.standard_commands_count,self.full_commands_count):
            if self.pattern_success_count[i]>4:
                pattern_list.append(self.pattern_list[i])
                pattern_success_count=np.append(pattern_success_count,self.pattern_success_count[i])
                pattern_trial_count=np.append(pattern_trial_count,self.pattern_trial_count[i])
                full_commands_count+=1
        self.pattern_list = pattern_list
        self.pattern_trial_count = pattern_trial_count
        self.pattern_success_count = pattern_success_count
        self.full_commands_count=full_commands_count
        self.to_file()
            
    def show(self):
        for i in range(self.full_commands_count):
            print(i,self.pattern_list[i],self.pattern_success_count[i],self.pattern_trial_count[i],
                                                                self.pattern_success_count[i]/self.pattern_trial_count[i])

def episodes_to_list(episodes):
    #на входе лист вида [[1:10],[20:50],...]
    #на выходе лист вида: [1,2,3,4,5,6,7,8,9,20,21,22,23,...]
    outlist=[]
    for epi in episodes:
        outlist+=range(epi[0],epi[1])
    return outlist


if 0:
    class multy_lgb(object):
        def __init__(self,lgbparams):
            self.lgbparams=lgbparams
            return

        def fit(self,X,Y):
            #Y - только np, иначе - дорабатывай код.
            y_width=np.shape(Y)[1]
            self.boost_list=[]
            l=int(0.75*X.shape[0])
            for i in range(y_width):
                train_data = lgb.Dataset(X[:l,:], Y[:l,i])
                eval_data = lgb.Dataset(X[l:,:], Y[l:,i])
                l_this = lgb.train(self.lgbparams,
                            train_data,
                            valid_sets=eval_data,
                            num_boost_round=5000,
                            early_stopping_rounds=2,
                            verbose_eval=False)
                self.boost_list.append(l_this)
            return

        def predict(self,X):
            lst_out = []
            for i in range(len(self.boost_list)):
                lst_out.append(self.boost_list[i].predict(X))
            return np.vstack(lst_out).T

class multy_xgb(object):
    def __init__(self,xgbparams,linear_out=True):
        self.linear_out=linear_out
        self.xgbparams=xgbparams
        return

    def fit(self,X,Y):
        #Y - только np, иначе - дорабатывай код.
        y_width=np.shape(Y)[1]
        self.boost_list=[]
        l=int(0.75*X.shape[0])
        for i in range(y_width):
            l_this=xgb.XGBRegressor(**self.xgbparams)
            l_this.fit(X[:l,:], Y[:l,i],
                       eval_set=[(X[l:,:], Y[l:,i])],
                       verbose=False)
            self.boost_list.append(l_this)
        if self.linear_out:
            Y_pred = self.predict(X,linear_use=False)
            linear_input = np.hstack((Y_pred,X))
            self.lm = linear_model.Ridge(alpha=0.1,max_iter=5000,random_state=0,normalize=True).fit(linear_input, Y)
        return
            
    def predict(self,X,linear_use=True):
        #linear_use - использовать ли линейный постобработчик, если он есть
        lst_out = []
        for i in range(len(self.boost_list)):
            lst_out.append(self.boost_list[i].predict(X))
        Y_pred = np.vstack(lst_out).T
        if linear_use and self.linear_out:
            linear_input = np.hstack((Y_pred,X))
            linear_input[np.isnan(linear_input)] = 0 #убрать nanы
            linear_input[linear_input>1e20] = 1e20 #убрать бесконечности
            linear_input[linear_input<-1e20] = -1e20
            Y_pred = self.lm.predict(linear_input)
        return Y_pred


def challenge_supervised(l):
    measure = False
    if measure:
        nw = pd.Timestamp.now()
    code,episodes,X,Y,steps_per_x,regularization,size_mem,pl = l
    code_len=len(code)
    if type(code) !=type(np.array([1,2,3,4])):
        #не взлетело
        return np.inf
    #steps_per_x = 50 #надо побольше, так как у нас тут язык не очень эффективный
    err_arr = []
    Y_pred = []
    sym = symbolic_regression(tact_count=steps_per_x,memory_size=size_mem,zero_dup=1,
                              size_genom=20,regularization=regularization,pl=pl)
    sym.genom=code
    sym.X_size = X.shape[1]
    sym.Y_size = Y.shape[1]
    Y_pred = sym.predict_raw(X,episodes)
    linear_input = np.hstack((X[episodes_to_list(episodes),:],Y_pred.T))
    try:
        if sym.postprocessing_learn=='xgbl':
            params = {
                    'booster':'gbtree',
                    'metric':'mse',
                    #'objective':'reg:squarederror',
                    'verbosity':0,
                    'max_depth': 6,
                    'n_estimators': 18,
                    'eta': 0.5,
                    'nthreads': 2,
                    'seed':0
                }
            lm = multy_xgb(params)
        elif sym.postprocessing_learn=='lin':
            lm = linear_model.Ridge(alpha=0.1,max_iter=16,random_state=0).fit(linear_input, Y[episodes_to_list(episodes),:])
        elif sym.postprocessing_learn=='raw':
            lm = Y_pred[:,:Y_size].T
        
        if sym.postprocessing_learn!='raw':
            lm.fit(linear_input, Y[episodes_to_list(episodes),:])
    except Exception as e:
        print('AAAAAAAAAAAAAAa',str(e))
        return np.inf
    Y_pred_lm = lm.predict(linear_input)
    Y_pred_lm_normed = Y_pred_lm/(np.abs(Y[episodes_to_list(episodes),:]).mean(axis=0)+1e-20)
    Y_normed = Y[episodes_to_list(episodes),:]/(np.abs(Y[episodes_to_list(episodes),:]).mean(axis=0)+1e-20)
    err = (Y_pred_lm_normed - Y_normed)**2
    data_len = err.shape[0]
    mse = np.mean(err)
    #sse=np.sum(err)
    #сумма от квадратов ошибок плюс длина кода на регуляризационный коэффициент
    #деление - просто чтобы сравнивать было легче
    #масштабирование нужно только чтобы с 1 сравнивать в поиске паттернов
    code = np.array(pl.compilation(code))
    count_code_blocks=np.sum(code>=0)
    mean_code_blocks=count_code_blocks/code_len
    if measure:
        print('challenge_time',pd.Timestamp.now() - nw,flush=True)  
    #2 компоненты: 
    #mse/np.mean(Y_normed**2) - log(P(модель|данные)), считаем, что модель прогнозирует матожидание, а результат должен быть распределён по Гауссу
    #mean_code_blocks/data_len - вклад от априорной вероятности модели
    loss = mse/np.mean(Y_normed**2)  + regularization * mean_code_blocks/data_len
    return loss
#+ regularization*np.sum(code>=0)

class automat:
    def __init__(self,code,parent,verbose):
        self.pos = [0,0,0]
        self.code_pos=0
        self.code=np.round(code)
        self.code = self.code[self.code>-1]
        self.buf1=0
        self.buf2=1
        self.cycle_opens=[]
        self.verbose=verbose
        self.copymode = 0
        self.parent = parent
        self.adr_belt = 2
    def exe(self,cmd):
        parent = self.parent
        try:
            if cmd==-1:#next command
                if self.verbose:
                    print(f'pass ')
                return
            elif cmd==0:
                #do nothing
                pass
            for i in range(3):
                if cmd==1+0+i*6:#move+
                    self.pos[i] +=1
                    self.pos[i] = self.pos[i]%parent.tab_len[i]
                    if self.verbose:
                        print(f'mv{i}+ {self.pos[i]}')
                elif cmd==1+1+i*6:#move-
                    self.pos[i] -=1
                    if self.pos[i]<0:
                        self.pos[i] += parent.tab_len[i]
                    if self.verbose:
                        print(f'mv{i}- {self.pos[i]}')
                elif cmd==1+2+i*6:#++
                    parent.tab[i][self.pos[i]]+=1
                    if self.verbose:
                        print(f'{i}++ {parent.tab[i][self.pos[i]]}')
                elif cmd==1+3+i*6:#--
                    parent.tab[i][self.pos[i]]-=1
                    if self.verbose:
                        print(f'{i}-- {parent.tab[i][self.pos[i]]}')
                elif cmd==1+4+i*6: #r
                    self.buf1=parent.tab[i][self.pos[i]]
                    if self.verbose:
                        print(f'r{i} buf={self.buf1}')
                elif cmd==1+5+i*6: #w
                    parent.tab[i][self.pos[i]]=self.buf1
                    if self.verbose:
                        print(f'w{i} buf={self.buf1}')
            #5+2*6=5+12=17
            if cmd==19:
                #cycle open
                self.cycle_opens.append(self.code_pos)
                if self.verbose:
                    print(f'[ ')
            elif cmd==20:
                #cycle close
                if len(self.cycle_opens)==0:
                    if self.verbose:
                        print(f']err ')
                    return
                if self.buf1>0:
                    self.code_pos=self.cycle_opens[-1]+1#return
                    if self.verbose:
                        print(f']<- ')
                else:
                    #through
                    self.cycle_opens.pop()
                    if self.verbose:
                        print(f']-> ')

            elif cmd==21:
                #execute
                if not (self.buf1 in [20, 21]):
                    self.exe(self.buf1)
                    if self.verbose:
                        print(f'exec ')
            elif cmd==22:
                #sort mem3,buf
                if parent.tab[2][self.pos[2]]>self.buf1:
                    buf = self.buf1
                    self.buf1 = parent.tab[2][self.pos[2]]
                    parent.tab[2][self.pos[2]] = buf
                    if self.verbose:
                        print(f'sort+ ')
                elif self.verbose:
                    print(f'sort- ')
            elif cmd==23:
                #swap mem3,buf
                buf = self.buf1
                self.buf1 = parent.tab[2][self.pos[2]]
                parent.tab[2][self.pos[2]] = buf
                if self.verbose:
                    print(f'swp')
            elif cmd==24:
                #sort in place
                if self.pos[2]<len(parent.tab[2])-1:
                    parent.tab[2][self.pos[2]:self.pos[2]+2] = np.sort(parent.tab[2][self.pos[2]:self.pos[2]+2])
                    if self.verbose:
                        print(f'sort_place+')
                else:
                    if self.verbose:
                        print(f'sort_place-')
            elif cmd==25:
                #cp 0->2
                #копируем buf1 символов. Частота равна buf2
                period=np.round(self.buf2)
                period = int(abs(round(period)))
                if period==0:
                    period=1
                end_copy=self.pos[0]+self.buf1*period
                if end_copy>=len(parent.tab[0]):
                    end_copy=len(parent.tab[0])-1
                start_copy=self.pos[0]
                what_to_copy=parent.tab[0][start_copy:end_copy]
                what_to_copy = what_to_copy[range(0,len(what_to_copy),period)]
                
                end_target=min(len(what_to_copy)+self.pos[2],len(parent.tab[2]))
                len_target=end_target-self.pos[2]
                what_to_copy = what_to_copy[:len_target]
                if self.copymode==0:
                    #parent.tab[2][self.pos[2]]=parent.tab[0][self.pos[0]]
                    parent.tab[2][self.pos[2]:self.pos[2]+len_target]=what_to_copy
                elif self.copymode==1:
                    parent.tab[2][self.pos[2]:self.pos[2]+len_target]+=what_to_copy
                elif self.copymode==2:
                    parent.tab[2][self.pos[2]:self.pos[2]+len_target]*=what_to_copy
                elif self.copymode==3:
                    parent.tab[2][self.pos[2]]=(parent.tab[2][self.pos[2]]+parent.tab[0][self.pos[0]])*0.5
                    parent.tab[2][self.pos[2]:self.pos[2]+len_target]=(what_to_copy+
                                                                       parent.tab[2][self.pos[2]:self.pos[2]+len_target])*0.5
                self.pos[0] +=len_target*period
                self.pos[0] = self.pos[0]%parent.tab_len[0]
                self.pos[2] +=len_target
                self.pos[2] = self.pos[2]%parent.tab_len[2]
                if self.verbose:
                    print(f'cp0-2')
            elif cmd==26:
                #cp 2->1
                if self.copymode==0:
                    parent.tab[1][self.pos[1]]=parent.tab[2][self.pos[2]]
                elif self.copymode==1:
                    parent.tab[1][self.pos[1]]+=parent.tab[2][self.pos[2]]
                elif self.copymode==2:
                    parent.tab[1][self.pos[1]]*=parent.tab[2][self.pos[2]]
                elif self.copymode==3:
                    parent.tab[1][self.pos[1]]=(parent.tab[1][self.pos[1]]+parent.tab[2][self.pos[2]])*0.5
                self.pos[1] +=1
                self.pos[1] = self.pos[1]%parent.tab_len[1]
                self.pos[2] +=1
                self.pos[2] = self.pos[2]%parent.tab_len[2]
                if self.verbose:
                    print(f'cp2-1')
            elif cmd==27:
                #copymode +
                self.copymode = 1
                if self.verbose:
                    print(f'cpmode+')
            elif cmd==28:
                #copymode *
                self.copymode = 2
                if self.verbose:
                    print(f'cpmode*')
            elif cmd==29:
                #copymode only copy
                self.copymode = 0
                if self.verbose:
                    print(f'cpmode cp')
            elif cmd==30:
                #абсолютная адресация
                if np.isinf(self.buf1) or np.isnan(self.buf1):
                    self.buf1 = 0
                self.pos[self.adr_belt] = int(self.buf1)%parent.tab_len[self.adr_belt]
                if self.pos[self.adr_belt] < 0:
                    self.pos[self.adr_belt] += parent.tab_len[self.adr_belt]
                if self.verbose:
                    print(f'pos{self.adr_belt}absto{self.pos[self.adr_belt]}')
            elif cmd==31:
                #относительная адресация
                if np.isinf(self.buf1):
                    self.buf1 = 0
                arg = self.buf1 + self.pos[self.adr_belt]
                if np.isnan(arg):
                    arg = 0
                self.pos[self.adr_belt] = int(arg)%parent.tab_len[self.adr_belt]
                if self.pos[self.adr_belt] < 0:
                    self.pos[self.adr_belt] += parent.tab_len[self.adr_belt]
                if self.verbose:
                    print(f'pos{self.adr_belt}relto{self.pos[self.adr_belt]}')
            elif cmd==32:
                #создать бота
                code_len = self.buf1
                start_point = int(self.pos[2] + 1)
                arg = self.pos[2] + 1 + code_len
                if abs(arg)>1e10 or np.isnan(arg):
                    arg = 0
                end_point = min([int(arg), parent.tab_len[2]])
                if start_point < parent.tab_len[2]:
                    code_cur = parent.tab[2][start_point:end_point]
                    code_cur = code_cur[code_cur>-1]
                    if len(code_cur)>3:
                        automat_new = automat(code_cur,parent,self.verbose)
                        arg = parent.tab[2][self.pos[2]]
                        if np.isinf(arg) or np.isnan(arg):
                            arg = 0
                        automat_new.pos[2] = int(arg)%parent.tab_len[2]#
                        if automat_new.pos[2] < 0:
                            automat_new.pos[2] += parent.tab_len[2]
                        parent.automat_bundle.append(automat_new)
                        if self.verbose:
                            print(f'bot №{len(parent.automat_bundle)-1} created')
                    else:
                        if self.verbose:
                            print(f'err bot creation')
                else:
                    if self.verbose:
                        print(f'err bot creation')
            elif cmd==33:
                #удалить бота
                if len(parent.automat_bundle)>1:
                    parent.automat_bundle = parent.automat_bundle[:-1]
                    if self.verbose:
                        ln = len(parent.automat_bundle) + 1
                        print(f'bot {ln} deleted')
                else:
                    if self.verbose:
                        print(f'err bot delete')
            elif cmd==34:
                #1/buf
                try:
                    self.buf1 = 1./self.buf1
                    if self.verbose:
                        print(f'1/buf')
                except Exception:
                    if self.verbose:
                        print(f'Error 1/buf')
            elif cmd==35:
                #return
                if self.verbose:
                    print(f'ret')
            elif cmd==36:
                #copymode mean
                self.copymode = 3
                if self.verbose:
                    print(f'cpmode mean')
            elif cmd==37:
                #adresation belt to 0
                self.adr_belt = 0
                if self.verbose:
                    print(f'chbelt 0')
            elif cmd==38:
                #adresation belt to 1
                self.adr_belt = 1
                if self.verbose:
                    print(f'chbelt 1')
            elif cmd==39:
                #adresation belt to 2
                self.adr_belt = 2
                if self.verbose:
                    print(f'chbelt 2')
            elif cmd==40:
                #copy buf1 to buf2
                self.buf2=self.buf1
                if self.verbose:
                    print(f'cpb1-b2')
            elif cmd==41:
                self.pos[0] = 0
                if self.verbose:
                    print(f'belt0 restart')
            elif cmd==42:
                self.pos[1] = 0
                if self.verbose:
                    print(f'belt1 restart')
            elif cmd==43:
                self.pos[2] = 0
                if self.verbose:
                    print(f'belt2 restart')
            elif cmd==44:
                #копируем buf1 символов. Частота равна buf2
                #Но все копии сваливаем в одну ячейку
                period=np.round(self.buf2)
                period = int(abs(round(period)))
                if period==0:
                    period=1
                end_copy=self.pos[0]+self.buf1*period
                if end_copy>=len(parent.tab[0]):
                    end_copy=len(parent.tab[0])-1
                start_copy=self.pos[0]
                what_to_copy=parent.tab[0][start_copy:end_copy]
                what_to_copy = what_to_copy[range(0,len(what_to_copy),period)]
                
                end_target=min(len(what_to_copy)+self.pos[2],len(parent.tab[2]))
                len_target=end_target-self.pos[2]
                what_to_copy = what_to_copy[:len_target]
                what_to_copy = np.mean(what_to_copy)
                if self.copymode==0:
                    parent.tab[2][self.pos[2]]=what_to_copy
                elif self.copymode==1:
                    parent.tab[2][self.pos[2]]+=what_to_copy
                elif self.copymode==2:
                    parent.tab[2][self.pos[2]]*=what_to_copy
                elif self.copymode==3:
                    parent.tab[2][self.pos[2]]=(parent.tab[2][self.pos[2]]+parent.tab[0][self.pos[0]])*0.5
                    parent.tab[2][self.pos[2]]=(what_to_copy+parent.tab[2]
                                                [self.pos[2]:self.pos[2]+len_target])*0.5
                if self.verbose:
                    print(f'cp0-2 convolve')
            elif cmd==45:
                #копируем buf1 символов. Частота равна buf2
                #Но все копии сваливаем в одну ячейку
                period=np.round(self.buf2)
                period = int(abs(round(period)))
                if period==0:
                    period=1
                end_copy=self.pos[2]+self.buf1*period
                if end_copy>=len(parent.tab[0]):
                    end_copy=len(parent.tab[0])-1
                start_copy=self.pos[2]
                what_to_copy=parent.tab[2][start_copy:end_copy]
                what_to_copy = what_to_copy[range(0,len(what_to_copy),period)]
                
                end_target=min(len(what_to_copy)+self.pos[2],len(parent.tab[2]))
                len_target=end_target-self.pos[2]
                what_to_copy = what_to_copy[:len_target]
                what_to_copy = np.mean(what_to_copy)
                if self.copymode==0:
                    parent.tab[2][self.pos[2]]=what_to_copy
                elif self.copymode==1:
                    parent.tab[2][self.pos[2]]+=what_to_copy
                elif self.copymode==2:
                    parent.tab[2][self.pos[2]]*=what_to_copy
                elif self.copymode==3:
                    parent.tab[2][self.pos[2]]=(parent.tab[2][self.pos[2]]+parent.tab[0][self.pos[0]])*0.5
                    parent.tab[2][self.pos[2]]=(what_to_copy+parent.tab[2]
                                                [self.pos[2]:self.pos[2]+len_target])*0.5
                if self.verbose:
                    print(f'cp2-2 convolve')
            elif cmd==46:
                #взвешенная сумма 0->2
                #взять из ячейки cell[arg1]*arg3+cell[arg2]*arg4, сложить и положить в ячейку арг5 (как предписывает копирование)
                count_args = 5
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'wsum02 error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    try:
                        dest_belt = 2
                        start_belt = 0
                        what_to_copy = parent.tab[start_belt][int(args[0])]*args[1] + parent.tab[start_belt][int(args[2])]*args[3]
                        if self.copymode==0:
                            parent.tab[dest_belt][int(args[5])]=what_to_copy
                        elif self.copymode==1:
                            parent.tab[dest_belt][int(args[5])]+=what_to_copy
                        elif self.copymode==2:
                            parent.tab[dest_belt][int(args[5])]*=what_to_copy
                        elif self.copymode==3:
                            parent.tab[dest_belt][int(args[5])]=(parent.tab[dest_belt][int(args[5])]+what_to_copy)*0.5
                        self.automat_bundle[automat_pointer].code_pos+=count_args
                        if self.verbose:
                            print(f'wsum02')
                    except Exception:
                        if self.verbose:
                            print(f'wsum02 error')
            elif cmd==47:
                #взвешенная сумма 2->2
                #взять из ячейки cell[arg1]*arg3+cell[arg2]*arg4, сложить и положить в ячейку арг5 (как предписывает копирование)
                count_args = 5
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'wsum22 error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    try:
                        dest_belt = 2
                        start_belt = 2
                        what_to_copy = parent.tab[start_belt][int(args[0])]*args[1] + parent.tab[start_belt][int(args[2])]*args[3]
                        if self.copymode==0:
                            parent.tab[dest_belt][int(args[5])]=what_to_copy
                        elif self.copymode==1:
                            parent.tab[dest_belt][int(args[5])]+=what_to_copy
                        elif self.copymode==2:
                            parent.tab[dest_belt][int(args[5])]*=what_to_copy
                        elif self.copymode==3:
                            parent.tab[dest_belt][int(args[5])]=(parent.tab[dest_belt][int(args[5])]+what_to_copy)*0.5
                        self.automat_bundle[automat_pointer].code_pos+=count_args
                        if self.verbose:
                            print(f'wsum22')
                    except Exception:
                        if self.verbose:
                            print(f'wsum22 error')
            elif cmd==48:
                #взвешенная разность модуль 0->2
                #взять из ячейки cell[arg1]*arg3+cell[arg2]*arg4, сложить и положить в ячейку арг5 (как предписывает копирование)
                count_args = 5
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'wdiv02 error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    try:
                        dest_belt = 2
                        start_belt = 0
                        what_to_copy = abs(parent.tab[start_belt][int(args[0])]*args[1] - parent.tab[start_belt])[int(args[2])]*args[3]
                        if self.copymode==0:
                            parent.tab[dest_belt][int(args[5])]=what_to_copy
                        elif self.copymode==1:
                            parent.tab[dest_belt][int(args[5])]+=what_to_copy
                        elif self.copymode==2:
                            parent.tab[dest_belt][int(args[5])]*=what_to_copy
                        elif self.copymode==3:
                            parent.tab[dest_belt][int(args[5])]=(parent.tab[dest_belt][int(args[5])]+what_to_copy)*0.5
                        self.automat_bundle[automat_pointer].code_pos+=count_args
                        if self.verbose:
                            print(f'wdiv02')
                    except Exception:
                        if self.verbose:
                            print(f'wdiv02 error')
            elif cmd==49:
                #взвешенная разность модуль 2->2
                #взять из ячейки cell[arg1]*arg3+cell[arg2]*arg4, сложить и положить в ячейку арг5 (как предписывает копирование)
                count_args = 5
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'wsdiv22 error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    try:
                        dest_belt = 2
                        start_belt = 2
                        what_to_copy = abs(parent.tab[start_belt][int(args[0])]*args[1] - parent.tab[start_belt][int(args[2])]*args[3])
                        if self.copymode==0:
                            parent.tab[dest_belt][int(args[5])]=what_to_copy
                        elif self.copymode==1:
                            parent.tab[dest_belt][int(args[5])]+=what_to_copy
                        elif self.copymode==2:
                            parent.tab[dest_belt][int(args[5])]*=what_to_copy
                        elif self.copymode==3:
                            parent.tab[dest_belt][int(args[5])]=(parent.tab[dest_belt][int(args[5])]+what_to_copy)*0.5
                        self.automat_bundle[automat_pointer].code_pos+=count_args
                        if self.verbose:
                            print(f'wdiv22')
                    except Exception:
                        if self.verbose:
                            print(f'wdiv22 error')
            elif cmd==50:
                if self.verbose:
                    print(f'if close')
                d1 = abs(parent.tab[0][7]-parent.tab[0][5])
                d2 = abs(parent.tab[0][9]-parent.tab[0][5])
                r1 = abs(parent.tab[0][8]-parent.tab[0][6])
                r2 = abs(parent.tab[0][10]-parent.tab[0][6])
                ans = 0
                if (d1<1.5) and (r1<45):
                    ans = 0.5
                if (d2<1.5) and (r2<45):
                    ans = 0.5
                if (d1<1) and (r1<20):
                    ans = 1
                if (d2<1) and (r2<20):
                    ans = 1
                parent.tab[2][len(parent.tab[2]) - 1]=ans
            elif cmd==51:
                if self.verbose:
                    print(f'predict polynomic')
                [tm,proj_1_r,proj_1_dir,proj_2_r,proj_2_dir,cooldown,angle,plane_dir,delta_dir,plane_r,cw,c_shoot,reward] = parent.tab[0][:13]
                x_plane = plane_r*np.cos(plane_dir*3.141/180)
                y_plane = plane_r*np.sin(plane_dir*3.141/180)
                proj1_x = proj_1_r*np.cos(proj_1_dir*3.141/180)
                proj1_y = proj_1_r*np.sin(proj_1_dir*3.141/180)
                parent.tab[2][8:12] = parent.tab[2][4:8] #очень старые значения
                parent.tab[2][4:8] = parent.tab[2][:4] #старые значения
                parent.tab[2][:4] = np.array([x_plane,y_plane,proj1_x,proj1_y]) #новые значения
                vect_now = np.array([x_plane, y_plane, proj1_x, proj1_y])
                d11 = parent.tab[2][:4] - parent.tab[2][4:8]
                d12 = parent.tab[2][4:8] - parent.tab[2][8:12]
                d2 = d11 - d12
                vect_predict = vect_now + d11 + d2*0.5
                proj_1_r = np.sqrt(proj1_x**2 + proj1_y**2)
                proj_1_dir = np.arctan2(proj1_y, proj1_x)*180/3.141
                plane_r = np.sqrt(x_plane**2 + y_plane**2)
                plane_dir = np.arctan2(y_plane, x_plane)*180/3.141
                parent.tab[2][12:16] = np.array([plane_r,plane_dir,proj_1_r,proj_1_dir])
                parent.tab[2][16] = (proj1_x - x_plane)**2 + (proj1_y - y_plane)**2 #-reward?
            elif cmd==52:
                if self.verbose:
                    print(f'ret_begin')
                self.code_pos = 0
            elif cmd==53:
                #0я лента переход в ячейку с абсолютным номером
                count_args = 1
                belt = 0
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'abs mov {belt} error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    try:
                        if (parent.tab_len[belt]>int(args[0])) and (int(args[0])>=0):
                            self.pos[belt] = int(args[0])
                            if self.verbose:
                                print(f'abs mov {belt}')
                        else:
                            if self.verbose:
                                print(f'abs mov {belt} error')
                    except Exception:
                        if self.verbose:
                            print(f'abs mov {belt} error')
            elif cmd==54:
                #1я лента переход в ячейку с абсолютным номером
                count_args = 1
                belt = 1
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'abs mov {belt} error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    try:
                        if (parent.tab_len[belt]>int(args[0])) and (int(args[0])>=0):
                            self.pos[belt] = int(args[0])
                            if self.verbose:
                                print(f'abs mov {belt}')
                        else:
                            if self.verbose:
                                print(f'abs mov {belt} error')
                    except Exception:
                        if self.verbose:
                            print(f'abs mov {belt} error')
            elif cmd==55:
                #2я лента переход в ячейку с абсолютным номером
                count_args = 1
                belt = 2
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'abs mov {belt} error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    try:
                        if (parent.tab_len[belt]>int(args[0])) and (int(args[0])>=0):
                            self.pos[belt] = int(args[0])
                            if self.verbose:
                                print(f'abs mov {belt}')
                        else:
                            if self.verbose:
                                print(f'abs mov {belt} error')
                    except Exception:
                        if self.verbose:
                            print(f'abs mov {belt} error')
            elif cmd==56:
                #экстраполяция
                count_args = 2
                belt = 2
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'extrapolation error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    source = int(args[0])
                    destination = int(args[1])
                    try:
                        source_val = parent.tab[belt][source]
                        destination_val = parent.tab[belt][destination]
                        der = source_val - destination_val
                        predict = source_val + der
                        parent.tab[belt][destination] = predict
                        parent.tab[belt][destination + 1] = der
                        if self.verbose:
                            print(f'extrapolation')
                    except Exception:
                        if self.verbose:
                            print(f'extrapolation error')
            elif cmd==57:
                #экспонентциальное забывание
                count_args = 3
                belt = 2
                if self.code_pos+1+count_args>=self.code:
                    if self.verbose:
                        print(f'exponent_filter error')
                else:
                    args = self.code[self.code_pos+1:self.code_pos+1+count_args]
                    source = int(args[0])
                    destination = int(args[1])
                    n = args[2]
                    try:
                        source_val = parent.tab[belt][source]
                        destination_val = parent.tab[belt][destination]
                        parent.tab[belt][destination] = (destination_val + source_val*(n-1))/n
                        if self.verbose:
                            print(f'exponent_filter')
                    except Exception:
                        if self.verbose:
                            print(f'exponent_filter error')
            
            
        except Exception:
            if self.verbose:
                print('error in exec')
        return


class bfBot:
    def __init__(self,code,pl,verbose=False,sizes=[8,10,200]):
        self.tab_len = sizes #in,out,mem
        self.tab=[np.zeros(self.tab_len[0]),np.zeros(self.tab_len[1]),np.zeros(self.tab_len[2])]
        self.verbose = verbose
        code=pl.compilation(code)
        self.automat_bundle = [automat(code,self,verbose)]
        self.pl=pl
    def execute_bundle(self,tacts):
        try:
            automat_pointer = 0
            while tacts>0:
                tacts -= 1
                if len(self.automat_bundle[automat_pointer].code)>0:
                    cmd = self.automat_bundle[automat_pointer].code[self.automat_bundle[automat_pointer].code_pos]
                    self.automat_bundle[automat_pointer].exe(cmd)
                    self.tab[1][np.isnan(self.tab[1])] = 0
                    self.tab[1][np.isinf(self.tab[1])] = 0
                    if cmd==34:
                        tacts = 0
                        #return
                if automat_pointer<len(self.automat_bundle):
                    self.automat_bundle[automat_pointer].code_pos+=1
                    if self.automat_bundle[automat_pointer].code_pos>=len(self.automat_bundle[automat_pointer].code):
                        self.automat_bundle[automat_pointer].code_pos=0
                automat_pointer+=1
                if automat_pointer>=len(self.automat_bundle):
                    automat_pointer = 0
        except Exception():
            #При отладке убирать
            pass
                
def disassemble(m_code):
    m_code = np.round(m_code)
    m_code=m_code[m_code>-1]
    print(m_code)
    lst = [cmd_names[int(i)] if int(i) in cmd_names else '|_'+str(i)+'\n'+disassemble(pl.pattern_list[int(np.round(i))])+'\n_|'  for i in m_code]
    s = ''''''
    bias=''
    for el in lst:
        if el=='_':
            continue
        if el=='[':
            bias+='    '
        if el==']':
            s+=bias+el+'\n'
            if len(bias)>=4:
                bias=bias[:-3]
        if el!='' and el!=']':
            s+=bias+el+'\n'
    return s

cmd_names ={
-1: '',
0:'',
1: 'mv0+',
2: 'mv0-',
3: '0++',
4: '0--',
5: 'r0',
6: 'w0',
7: 'mv1+',
8: 'mv1-',
9: '1++',
10: '1--',
11: 'r1',
12: 'w1',
13: 'mv2+',
14: 'mv2-',
15: '2++',
16: '2--',
17: 'r2',
18: 'w2',
19: '[',
20: ']',
21: 'EXEC',
22: 'sort',
23: 'swp',
24: 'sort_place',
25: 'cp02',
26: 'cp21',
27:'cp_mode+',
28:'cp_mode*',
29:'cp_mode cp',
30:'absto',
31:'relto',
32:'crtbot',
33:'delbot',
34:'1/buf1',
35:'ret',
36:'cp_mode mean',
37:'chbelt 0',
38:'chbelt 1',
39:'chbelt 2',
40:'cpb1-b2',
41:'belt0restart',
42:'belt1restart',
43:'belt2restart',
44:'cp0-2 convolve',
45:'cp2-2 convolve',
46:'wsum02',
47:'wsum22',
48:'wdiv02',
49:'wdiv22',
50:'ifclose',
51:'predict_polynomic',
52:'ret_begin',
53:'abs mov 0',
54:'abs mov 1',
55:'abs mov 2',
56:'extrapolation',
57:'exp_filtration'
}

def evol_parallel(function,pl=None,bounds=[0,1],size_x=10,popsize=20,maxiter=10,mutation_p=0.1,mutation_p_e=0.01,
                  mutation_r=0.1,alpha_count=3,elitarism=2,n_jobs=1,seed=1,verbose=True,
                  out=[],episodes=None,X=None,Y=None,regularization=0.15,steps=30,size_mem=200,
                  start_point=[],get_extended=False,elementary_command_count=40):
    #steps - число тактов на один прогноз
    #ищем минимум
    
    #edge
    
    np.random.seed(seed)
    
    if pl is None:
        x_old = [np.random.random(size=size_x)*(bounds[1]-bounds[0]) + bounds[0] for i in range(popsize)]
    else:
        #конструкция для генерации геномов
        edge=elementary_command_count+1#граница между операторами и функциями
        trial=pl.pattern_trial_count.copy()
        trial[trial<1e2]=1e2
        trial[:edge]=pl.pattern_trial_count[:edge].copy()
        p_arr=(pl.pattern_success_count+1e-10)/(trial+100)
        p_arr[:edge][p_arr[:edge]<0.002]=0.002
        p_arr[0] = np.abs(bounds[0])/(bounds[1]-bounds[0])
        print('p_arr max,min,med,q0.25,q0.75',np.max(p_arr),np.min(p_arr),np.median(p_arr),
              np.percentile(p_arr,25),np.percentile(p_arr,75))
        #print(p_arr)
        x_old_old = [np.random.random(size=size_x)*(bounds[1]-bounds[0]) + bounds[0] for i in range(int(popsize/(edge-1)))]
        x_old=[np.array([float(np.argmax(np.random.rand(len(p_arr))*p_arr)) for i in range(size_x)]) for j in range(popsize-int(popsize/(edge-1))) ]
        x_old.extend(x_old_old)
        bounds = [0,(pl.full_commands_count-1)]
        for i in range(len(x_old)):
            x_old[i][np.round(np.array(x_old[i]))==0]=-1

    
    if len(start_point)>0:
        ln = np.min([len(start_point),len(x_old)])
        x_old[:ln]=start_point
   
    for t in range(maxiter):
        #if n_jobs == 1:
        if 0:
            if X is None:
                l_out = [function(x) for x in x_old]
            else:
                l_out = [[function([x,episodes,X,Y,steps,regularization,size_mem,pl])] for x in x_old]
        else:
            pool = Pool(processes=n_jobs)
            if X is None:
                l_out = pool.map(function, [x for x in x_old])
            else:
                l_out = pool.map(function, [[x,episodes,X,Y,steps,regularization,size_mem,pl] for x in x_old])
            pool.close()
            pool.join()   
        y_old = np.array(l_out)
        del l_out
       
        #отобрать альфачей
        alpha_nums = y_old.argsort()[:alpha_count]
           
        if verbose:
            print(f'iteration {t} y=',y_old[alpha_nums[:elitarism]])
        x_new = []
        for elit in range(elitarism):
            x_new.append(x_old[alpha_nums[elit]].copy())
        for child in range(popsize - elitarism):
            #скрещиваем
            crossed_alphas = alpha_nums[[np.random.randint(low=0,high=alpha_count),np.random.randint(low=0,high=alpha_count)]]
            x_c = x_old[alpha_nums[0]]
            idx = np.random.rand(len(x_c))<0.5
            x_c[idx] = x_old[alpha_nums[1]][idx]
            x_new.append(x_c)
            idx_muta = np.random.rand(len(x_c))<mutation_p
            x_c[idx_muta] += (np.random.rand(len(x_c[idx_muta]))-0.5)*2*mutation_r*(bounds[1]-bounds[0])
            x_c[x_c>bounds[1]]=bounds[1]
            x_c[x_c<bounds[0]]=bounds[0]
            x_new.append(x_c.copy())
        x_old = x_new
        if len(out)>0:
            out[0] = x_old.copy()
        mutation_p = mutation_p*(1-mutation_p_e)
    if 0:#n_jobs == 1:
        if X is None:
            l_out = [function(x) for x in x_old]
        else:
            l_out = [[function([x,episodes,X,Y,steps,regularization,size_mem,pl])] for x in x_old]
    else:
        pool = Pool(processes=n_jobs)
        if X is None:
            l_out = pool.map(function, [x for x in x_old])
        else:
            l_out = pool.map(function, [[x,episodes,X,Y,steps,regularization,size_mem,pl] for x in x_old])
        pool.close()
        pool.join()
    y_old = np.array(l_out)
    alpha_nums = y_old.argsort()[:alpha_count]
    if verbose:
        print('iteration final y=',y_old[alpha_nums[:elitarism]])
    code=x_new[alpha_nums[np.argmin(y_old[alpha_nums])]]
    print('NOCOMPILED',code[code>=0])
    print('COMPILED',pl.compilation(x_new[alpha_nums[np.argmin(y_old[alpha_nums])]]))
    if get_extended:
        return [x_new[alpha_nums[np.argmin(y_old[alpha_nums])]],x_old.copy(), np.array(l_out)]
    else:
        return x_new[alpha_nums[np.argmin(y_old[alpha_nums])]]
    
    
class symbolic_regression(object):
    def __init__(self,pl,memory_size=200,tact_count=10, zero_dup=2, size_genom=100,regularization=0.15,out=[1], postprocessing_prod='auto', postprocessing_learn='lin'):
        #memory_size - размер внутренней памяти в единичной машине Тьюринга
        #tact_count - число тактов на прогноз очередного значения
        #zero_dup - во сколько раз нулей больше, чем не-нулей
        #size_genom - длина генома
        self.memory_size = memory_size
        self.tact_count = tact_count
        self.zero_dup = zero_dup
        self.size_genom = size_genom
        self.out = out
        self.regularization = regularization
        self.lm = []
        self.X_size=0
        self.Y_size=0
        self.pl=pl
        #xgbl
        self.postprocessing_prod = postprocessing_prod
        self.postprocessing_learn = postprocessing_learn
    def fit(self,X,Y,episodes,popsize=450,maxiter=1000,mutation_p=0.1,mutation_p_e=0.1,
            mutation_r=1,alpha_count=28,elitarism=12,n_jobs=8,seed=0,verbose=True,start_point=[]):
        self.X_size=X.shape[1]
        self.Y_size=Y.shape[1]
        #episodes - это list с границами эпизодов обучения. [[0,10][10,25]]  
        #mutation_p_e - это штука, уменьшающая вероятность мутации. mutation_p_e=0.2 - уменьшаем каждый раз на 20%
        pl=pattern_lib()
        self.elementary_command_count=pl.standard_commands_count
        [self.genom,self.genom_list,self.profit_array] = evol_parallel(function=challenge_supervised, pl=pl, bounds=
                                                                       [int(-self.elementary_command_count*(self.zero_dup)),
                                                                        self.elementary_command_count],
                                                                       size_x=self.size_genom,
                                                                       popsize=popsize,maxiter=maxiter,mutation_p=mutation_p,
                                                                       mutation_p_e=mutation_p_e,mutation_r=mutation_r,
                                                                       alpha_count=alpha_count,elitarism=elitarism,
                                                                       n_jobs=n_jobs,seed=seed,
                                                                       verbose=verbose,out=self.out,episodes=episodes,X=X,
                                                                       Y=Y,regularization=self.regularization,
                                                                       steps=self.tact_count,
                                                                       size_mem=self.memory_size,
                                                                       start_point=start_point,
                                                                       get_extended=True,
                                                                       elementary_command_count=self.elementary_command_count)
        
        pl.update(self.genom_list, self.profit_array)
        pl.to_file()
        
        Y_pred = self.predict_raw(X,episodes)
        linear_input = np.hstack((X[episodes_to_list(episodes),:],Y_pred.T))
        #params = {'n_estimators': 9, 'base_estimator':DecisionTreeRegressor(max_depth=4),'loss': 'square','random_state':0}
        #self.lm = MultiOutputRegressor(ensemble.AdaBoostRegressor(**params))
        
        if self.postprocessing_prod=='xgbl':
            params = {
                    'booster':'gbtree',
                    'metric':'mse',
                    #'objective':'reg:squarederror',
                    'verbosity':0,
                    'max_depth': 7,
                    'n_estimators': 90,
                    'eta': 0.3,
                    'nthreads': 2,
                    'seed':0
                }
            self.lm = multy_xgb(params)
        elif self.postprocessing_prod=='lin':
            self.lm = linear_model.Ridge(alpha=50,max_iter=16,random_state=0).fit(linear_input, Y[episodes_to_list(episodes),:])
        elif self.postprocessing_prod=='raw':
            self.lm = Y_pred[:,:self.Y_size].T
        elif self.postprocessing_prod=='auto':
            list_mse = []
            lm_list = []
            params = {
                    'booster':'gbtree',
                    'metric':'mse',
                    #'objective':'reg:squarederror',
                    'verbosity':0,
                    'max_depth': 7,
                    'n_estimators': 70,
                    'eta': 0.3,
                    'nthreads': 2,
                    'seed':0,
                
                }
            lm_list.append(multy_xgb(params,linear_out=True))
            lm_list.append(multy_xgb(params,linear_out=False))
            lm_list.append(linear_model.Ridge(alpha=50,max_iter=16,random_state=0))
            lm_list.append(linear_model.Ridge(alpha=1000,max_iter=16,random_state=0))
            lm_list.append(linear_model.Ridge(alpha=1,max_iter=16,random_state=0))
            lm_list.append(linear_model.Ridge(alpha=0.05,max_iter=16,random_state=0))
            l = int(linear_input.shape[1]/2)
            for lm in lm_list:
                lm.fit(linear_input[:l,:], Y[episodes_to_list(episodes),:][:l,:])
                Y_pred_lm = lm.predict(linear_input[l:,:])
                Y_pred_lm_normed = Y_pred_lm/(np.abs(Y[episodes_to_list(episodes),:][l:,:]).mean(axis=0)+1e-20)
                Y_normed = Y[episodes_to_list(episodes),:][l:,:]/(np.abs(Y[episodes_to_list(episodes),:][l:,:]).mean(axis=0)+1e-20)
                err = (Y_pred_lm_normed - Y_normed)**2
                mse = np.mean(err)
                list_mse.append(mse)
            argmin = np.argmin(np.array(list_mse))
            if argmin==0:
                self.lm = multy_xgb(params,linear_out=True)
                print('selected boosting with linear layer, mse',list_mse[argmin],list_mse)
            elif argmin==1:
                self.lm = multy_xgb(params,linear_out=False)
                print('selected boosting without linear layer, mse',list_mse[argmin],list_mse)
            elif argmin==2:
                self.lm = linear_model.Ridge(alpha=50,max_iter=16,random_state=0).fit(linear_input, Y[episodes_to_list(episodes),:])
                print('selected linear model 50, mse',list_mse[argmin],list_mse)
            elif argmin==3:
                self.lm = linear_model.Ridge(alpha=1000,max_iter=16,random_state=0).fit(linear_input, Y[episodes_to_list(episodes),:])
                print('selected linear model 1000, mse',list_mse[argmin],list_mse)
            elif argmin==4:
                self.lm = linear_model.Ridge(alpha=1,max_iter=16,random_state=0).fit(linear_input, Y[episodes_to_list(episodes),:])
                print('selected linear model 1, mse',list_mse[argmin],list_mse)
            elif argmin==5:
                self.lm = linear_model.Ridge(alpha=1,max_iter=16,random_state=0).fit(linear_input, Y[episodes_to_list(episodes),:])
                print('selected linear model 0.05, mse',list_mse[argmin],list_mse)
            else:
                print('selected  WHAT THE HELL???')
                1/0
                
            
            
        #params = {'n_estimators': 7, 'max_depth': 7, 'min_samples_split': 2,'random_state':0}
        #self.lm = MultiOutputRegressor(ensemble.RandomForestRegressor(**params))
        if self.postprocessing_prod!='raw':
            self.lm.fit(linear_input, Y[episodes_to_list(episodes),:])
        
        #self.lm = linear_model.Ridge(alpha=10,max_iter=4,random_state=0).fit(linear_input, Y[episodes_to_list(episodes),:])
        return self.genom
    def predict_step(self,X,bot,i):
        x = X[i,:]
        bot.tab[0] = x.copy()
        bot.execute_bundle(self.tact_count)
        return np.append(bot.tab[1],bot.tab[2])
    def predict_raw(self,X,episodes=None,genom=None):
        if episodes is None:
            episodes = [[0,len(X)]]
        if genom is None:
            genom = self.genom
        pred = []
        for ep in episodes:
            bot = bfBot(genom.copy(),sizes=[self.X_size,self.Y_size,self.memory_size],pl=self.pl)
            pred_local = [self.predict_step(X,bot,i) for i in range(ep[0],ep[1])]
            pred.extend(pred_local)
            #for i in range(ep[0],ep[1]):
            #    x = X[i,:]
            #    bot.tab[0] = x.copy()
            #    bot.execute_bundle(self.tact_count)
            #    pred.append(np.append(bot.tab[1],bot.tab[2]))
        #тут хранится состояние бота
        self.current_bot=bot
        pred=np.array(pred).T
        pred[np.isnan(pred)] = 0 #убрать nanы
        pred[pred>1e20] = 1e20 #убрать бесконечности
        pred[pred<-1e20] = -1e20
        return pred
    def predict(self,X,episodes=None,genom=None):
        Y_pred = self.predict_raw(X,episodes,genom)
        linear_input = np.hstack((X[episodes_to_list(episodes),:],Y_pred.T))
        linear_input[np.isnan(linear_input)] = 0 #убрать nanы
        linear_input[linear_input>1e20] = 1e20 #убрать бесконечности
        linear_input[linear_input<-1e20] = -1e20
        
        Y_pred_lm = self.lm.predict(linear_input)
        return Y_pred_lm
    def predict_next(self,X):
        pred = []
        if type(X)==type(np.array([1])):
            X=np.array(X,ndmin=2)
        self.current_bot.execute_bundle(self.tact_count)
        pred.append(np.append(self.current_bot.tab[1],self.current_bot.tab[2]))    
        Y_pred=np.array(pred)
        Y_pred[np.isnan(Y_pred)]=0
        linear_input = np.hstack((X[:,:],Y_pred))
        Y_pred_lm = self.lm.predict(linear_input)
        return Y_pred_lm
    
class symbolic_regression_boosting(object):
    def __init__(self,memory_size=200,tact_count=10, zero_dup=2, size_genom=100,regularization=0.05, postprocessing_prod='auto', postprocessing_learn='lin'):
        #memory_size - размер внутренней памяти в единичной машине Тьюринга
        #tact_count - число тактов на прогноз очередного значения
        #zero_dup - во сколько раз нулей больше, чем не-нулей
        #size_genom - длина генома
        self.memory_size = memory_size
        self.tact_count = tact_count
        self.zero_dup = zero_dup
        self.size_genom = size_genom
        self.out = [1]
        self.regularization = regularization
        self.postprocessing_prod, self.postprocessing_learn = postprocessing_prod, postprocessing_learn
    def fit(self,X,Y,episodes,boosting_count=5,boosting_eta=0.2,popsize=450,maxiter=1000,mutation_p=0.1,
            mutation_p_e=0.1,mutation_r=1,alpha_count=28,elitarism=12,n_jobs=8,seed=0,t_index=None,verbose=True):
        self.X_size=X.shape[1]
        self.Y_size=Y.shape[1]
        self.t_index=t_index
        if not t_index is None:
            self.t_delta = X[-1,t_index]-X[-2,t_index]
        self.boosting_eta = boosting_eta
        self.boosting_count = boosting_count
        out=self.out
        #episodes - это list с границами эпизодов обучения. [[0,10][10,25]]  
        #mutation_p_e - это штука, уменьшающая вероятность мутации. mutation_p_e=0.2 - уменьшаем каждый раз на 20%
        self.genom = []
        self.boosting=[]
        i = 0
        pl = pattern_lib()
        while 1:
            sym = symbolic_regression(pl=pl,tact_count=self.tact_count,zero_dup=self.zero_dup,size_genom=self.size_genom,
                                      regularization=self.regularization, postprocessing_prod=self.postprocessing_prod, postprocessing_learn=self.postprocessing_learn)
            sym.fit(X,Y,episodes,maxiter=maxiter,popsize=popsize,mutation_p=mutation_p,mutation_r=mutation_r,seed=i, n_jobs=n_jobs)
            score = challenge_supervised([sym.genom,episodes,X,Y,self.tact_count,self.regularization,self.memory_size,pl])
            if score<1 or i>=4:
                if verbose:
                    print(f'BOOST 0 ready. Score {score}.i {i}')
                break
            else:
                i+=1
                if verbose:
                    print(f'BOOST 0 failed. Score {score}.i {i}')
            
        
        self.genom.append(sym.genom)
        self.boosting.append(sym)
        Y_pred = sym.predict(X,episodes=episodes)
        Y_new_target = Y.copy()
        Y_new_target[episodes_to_list(episodes),:] = Y[episodes_to_list(episodes),:] - Y_pred
        pl = pattern_lib()
        for i in range(1,boosting_count):
            sym = symbolic_regression(pl=pl,tact_count=self.tact_count,zero_dup=self.zero_dup,size_genom=self.size_genom,
                                      regularization=self.regularization)
            sym.fit(X,Y_new_target*boosting_eta,episodes,maxiter=maxiter,popsize=popsize,mutation_p=mutation_p,
                    mutation_r=mutation_r,seed=i)
            score = challenge_supervised([sym.genom,episodes,X,Y_new_target*boosting_eta,self.tact_count,
                                          self.regularization,self.memory_size,pl])
            if score<1:
                Y_pred = sym.predict(X,episodes=episodes)
                Y_new_target[episodes_to_list(episodes),:] = Y_new_target[episodes_to_list(episodes),:] - Y_pred
                self.genom.append(sym.genom)
                self.boosting.append(sym)
                if verbose:
                    print(f'BOOST {i} ready. Score:{score}')
            else:
                if verbose:
                    print(f'BOOST {i} failed. Score:{score}')
        
        return self.genom
    def predict(self,X,episodes=None):
        if episodes is None:
            episodes = [[0,len(X)]]
        
        pred_write = []
        for j in range(0,len(self.boosting)):
            pred=self.boosting[j].predict(X,episodes)       
            if len(pred_write)==0:
                pred_write = np.array(pred).T
            else:
                pred_write += np.array(pred).T
                
        return pred_write.T
    def predict_next(self,X):
        pred_write = []
        for j in range(0,len(self.boosting)):
            pred=self.boosting[j].predict_next(X)
            if not self.t_index is None:
                pred[0,self.t_index]=X[self.t_index]+self.t_delta
            if len(pred_write)==0:
                pred_write = np.array(pred).T
            else:
                pred_write += np.array(pred).T       
        return pred_write.T
    
    

class symbolic_regression_multyboosting(object):
    def __init__(self,memory_size=200,tact_count=10, zero_dup=2, size_genom=100,regularization=0.15,disco=1, postprocessing_prod='auto', postprocessing_learn='lin'):
        #memory_size - размер внутренней памяти в единичной машине Тьюринга
        #tact_count - число тактов на прогноз очередного значения
        #zero_dup - во сколько раз нулей больше, чем не-нулей
        #size_genom - длина генома
        self.memory_size = memory_size
        self.tact_count = tact_count
        self.zero_dup = zero_dup
        self.size_genom = size_genom
        self.out = [1]
        self.regularization = regularization #штраф за сложность модели
        self.disco=disco #затухание функции ошибки с ростом расстояния от начала прогноза
        self.postprocessing_prod, self.postprocessing_learn = postprocessing_prod, postprocessing_learn
    def fit(self,X,Y,episodes,forest_count=3,sample_part=0.5,boosting_count=5,boosting_eta=0.2,
            popsize=450,maxiter=1000,mutation_p=0.1,mutation_p_e=0.1,mutation_r=1,alpha_count=28,elitarism=12,
            n_jobs=8,seed=0,verbose=True, t_index=None):
        #X и Y могут быть не таблицами. а множествами таблиц, если на входе был выход мультимодели
        if type(X)==type([]):
            if type(Y)!=type([]):
                print('Warning! symbolic_regression_multyboosting.fit: histogram input, mono table out!')
            if len(X[0])!=forest_count or len(Y[0])!=forest_count:
                print('Warning! symbolic_regression_multyboosting.fit: len(X[0])!=forest_count or len(Y[0])!=forest_count!')
            histogram_input=True
        else:
            histogram_input=False
            
        self.forest=[]
        for i in range(forest_count):
            #сэмплируем
            np.random.seed(i*199+2)
            lst_ind=[int(j) for j in np.random.randint(0, len(episodes), int(np.ceil(len(episodes)*sample_part)))]
            episodes_current=np.array(episodes)[lst_ind]
            print(f'FOREST NUMBER {i}, episodes',episodes_current)
            sym=symbolic_regression_boosting(tact_count=self.tact_count,zero_dup=self.zero_dup,
                                             size_genom=self.size_genom,regularization=self.regularization,
                                             postprocessing_prod=self.postprocessing_prod, 
                                             postprocessing_learn=self.postprocessing_learn)
            if histogram_input==False:
                sym.fit(X=X,Y=Y,episodes=episodes_current,boosting_count=boosting_count,boosting_eta=boosting_eta,popsize=popsize,
                        maxiter=maxiter,mutation_p=mutation_p,mutation_p_e=mutation_p_e,mutation_r=mutation_r,
                        alpha_count=alpha_count,elitarism=elitarism,n_jobs=n_jobs,seed=seed+i*1000,
                        t_index=t_index,verbose=verbose)
                #рассчитать ошибку
                Y_pred_lm=sym.predict(X,episodes)
                Y_pred_lm_normed = Y_pred_lm/(np.abs(Y[episodes_to_list(episodes),:]).mean(axis=0)+1e-20)
                Y_normed = Y[episodes_to_list(episodes),:]/(np.abs(Y[episodes_to_list(episodes),:]).mean(axis=0)+1e-20)
                err = (Y_pred_lm_normed - Y_normed)**2
            else:
                #формат гистограммы: [pred_list,p_array,m_pred]
                sym.fit(X=X[0][i],Y=Y[0][i],episodes=episodes_current,boosting_count=boosting_count,boosting_eta=boosting_eta,
                        popsize=popsize,maxiter=maxiter,mutation_p=mutation_p,mutation_p_e=mutation_p_e,
                        mutation_r=mutation_r,alpha_count=alpha_count,elitarism=elitarism,
                        n_jobs=n_jobs,seed=seed+i*1000,t_index=t_index,verbose=verbose)
                #рассчитать ошибку
                Y_pred_lm=sym.predict(X[0][i],episodes)
                Y_pred_lm_normed = Y_pred_lm/(np.abs(Y[0][i][episodes_to_list(episodes),:]).mean(axis=0)+1e-20)
                Y_normed = Y[episodes_to_list(episodes),:]/(np.abs(Y[0][i][episodes_to_list(episodes),:]).mean(axis=0)+1e-20)
                err = (Y_pred_lm_normed - Y_normed)**2
            self.forest.append(sym)
            weight_line=(1./self.disco)**(np.array(range(err.shape[0])))
            weight_line=weight_line/np.mean(weight_line)
            err_augmented=err.copy()
            for j in range(err_augmented.shape[1]):
                err_augmented[:,j]=err_augmented[:,j]*weight_line
            
            #err_sum = 0.01*np.sum(err)/err.shape[1]
            err_sum = np.sum(err)
            complexity = sum([len(sym.boosting[j].genom[sym.boosting[j].genom>0]) for j in range(len(sym.boosting))])
            print('err_sum,complexity',err_sum,complexity)
            sym.log_likelyhood=err_sum+complexity
            
    def predict(self,X,episodes=None,return_hist=True):
        #X и Y могут быть не таблицами. а множествами таблиц, если на входе был выход мультимодели
        if type(X)==type([]):
            histogram_input=True
        else:
            histogram_input=False
        pred_list=[]
        for i in range(len(self.forest)):
            if not histogram_input:
                pred_list.append(self.forest[i].predict(X,episodes))
            else:
                pred_list.append(self.forest[i].predict(X[0][i],episodes))
        log_likelyhood_arr=np.array([sym.log_likelyhood for sym in self.forest])
        #log_likelyhood_arr-=min(log_likelyhood_arr)
        #print('log_likelyhood_arr',log_likelyhood_arr)
        log_likelyhood_arr=log_likelyhood_arr-np.min(log_likelyhood_arr)
        p_array=2**(-log_likelyhood_arr)
        if sum(p_array)>1e-50:
            p_array=p_array/sum(p_array)
        else:
            p_array=p_array*0+1/len(p_array)
        #print('p_array',p_array)
        #точечная оценка, матожидание
        m_pred = np.array([pred_list[i]*p_array[i] for i in range(len(pred_list))]).sum(axis=0)
        if return_hist:
            return [pred_list,p_array,m_pred]
        else:
            return m_pred
    def predict_next(self,X,return_hist=False):   
        #X и Y могут быть не таблицами. а множествами таблиц, если на входе был выход мультимодели
        if type(X)==type([]):
            histogram_input=True
        else:
            histogram_input=False

        pred_list=[]
        for i in range(len(self.forest)):
            if not histogram_input:
                pred_list.append(self.forest[i].predict_next(X))
            else:
                pred_list.append(self.forest[i].predict_next(X[0][i][0]))
            
        log_likelyhood_arr=np.array([sym.log_likelyhood for sym in self.forest])
        #log_likelyhood_arr-=min(log_likelyhood_arr)
        #print('log_likelyhood_arr',log_likelyhood_arr)
        log_likelyhood_arr=log_likelyhood_arr-np.min(log_likelyhood_arr)
        p_array=2**(-log_likelyhood_arr)
        if sum(p_array)>1e-50:
            p_array=p_array/sum(p_array)
        else:
            p_array=p_array*0+1/len(p_array)
        #print('p_array',p_array)
        #точечная оценка, матожидание
        m_pred = np.array([pred_list[i]*p_array[i] for i in range(len(pred_list))]).sum(axis=0)
        if return_hist:
            return [pred_list,p_array,m_pred]
        else:
            return m_pred
    #должна быть взвешенная сумма (с случае матожидания)
    #или вероятности (в случае гистограммы)
    #для этого надо хранить логарифм правдоподобия каждой модели
    #и обновлять по одиночным точкам - отдельная функция
#ПРОТОКОЛ ГИСТОГРАММЫ: [pred_list,p_array,m_pred]

class reality_model(object):
    #делаем reality_model.
    #На входе dataframe X с размеченными actions и reward
    #на данном этапе нам пофиг на множества возможных действий. One-hot? Рычаги? Ой да похрен.
    #методы: 
    #fit model. У нас может даже не быть эпизодов - мы их нарежем... Минимальная длина, максимальная длина, количество точек (в долях от всей истории).
    #one_step(actions=None) Если нет actions, то мы их прогнозируем. Причём каждая модель из forest действует независимо
    #  сохранить выходной результат
    #update_likelyhood(X_fact) - обновить правдоподобие. X_fact - это реальный кадр данных. А кадр прогноза - это то, что лежит в self.X
    #many_steps (actions=,steps_count=20,return_reward=False,disco_return=0.9) а дальше у нас list из dict из np_array и числа  [{ [10,-1,2],0 }, { [1,-1,2],2 }, { [1,-1,2],5 }  ] 
    #  dict: action, step_number
    #  return_reward - возвращать ли значение суммарного профита по этой траектории
    #init_prediction(tacts) - запустить модель на прогноз на данных, скажем, он -100 до -1 такта. 
    #  Смысл в том, чтобы поместить в память модели недавние события
    def __init__(self,memory_size=200,tact_count=10, zero_dup=2, 
                 size_genom=100,regularization=0.15,disco=1,action_fields=[],reward_field='reward',timedelta=1,timefield='tm'):
        self.memory_size=memory_size
        self.tact_count=tact_count
        self.zero_dup=zero_dup
        self.size_genom=size_genom
        self.regularization=regularization
        self.disco=disco
        self.action_fields=action_fields
        self.reward_field=reward_field
        self.df_ethalon=pd.DataFrame({1:[1,2],'2':[3,4]})
        self.timedelta=timedelta
        self.timefield=timefield
    def fit(self,X,episodes=None,episodes_max_len=100,episodes_min_len=10,episodes_percent=0.8,
            forest_count=3,sample_part=0.5,boosting_count=5,boosting_eta=0.2,popsize=450,maxiter=1000,mutation_p=0.1,
            mutation_p_e=0.1,mutation_r=1,alpha_count=28,elitarism=12,n_jobs=8,seed=0,verbose=True):
        self.means_X=X.abs().mean().values
        self.action_fields_where=[np.where(X.columns==field)[0][0] for field in self.action_fields]
        self.reward_field_where=np.where(X.columns==self.reward_field)[0][0]
        
        t_index=np.where(X.columns==self.timefield)[0]
        if len(t_index)==0:
            t_index=None
        else:
            t_index=t_index[0]
        
        if episodes is None:
            #значит, эпизоды надо нарезать
            episodes_sum_size=0
            xlen=X.shape[0]-1-self.timedelta
            episodes_sum_required=xlen*episodes_percent
            episodes=[]
            while episodes_sum_size<episodes_sum_required:
                episode_start=int(np.random.rand()*(xlen-episodes_min_len-1))
                if episode_start<0:
                    print('warning, episode_start=',episode_start)
                episode_len=int(np.random.rand()*(episodes_max_len-episodes_min_len)+episodes_min_len)
                episode_end=episode_start+episode_len
                if episode_end>=xlen-1:
                    episode_end=xlen-2
                episodes.append([episode_start,episode_end])
                episodes_sum_size+=episode_end-episode_start
        sym = symbolic_regression_multyboosting(memory_size=self.memory_size,tact_count=self.tact_count,zero_dup=self.zero_dup,
                                                size_genom=self.size_genom,regularization=self.regularization,disco=self.disco)
        X_fit=X[:-self.timedelta].values
        Y_fit=X[self.timedelta:].values
        self.forest_count=forest_count
        sym.fit(X_fit,Y_fit,episodes,forest_count=forest_count,sample_part=sample_part,
                boosting_count=boosting_count,maxiter=maxiter,popsize=popsize,mutation_p=mutation_p,mutation_r=mutation_r,
                n_jobs=n_jobs,alpha_count=alpha_count,elitarism=elitarism,seed=seed,verbose=verbose,t_index=t_index)
        self.forest=sym
    def init_prediction(self,X,horizon):
        X_pred=self.forest.predict(X[-horizon:].values)
        pred_list=[]
        p_list=[]
        for i in range(self.forest_count):
            pred_list.append(X[-self.timedelta:].values)
            p_list.append(1./self.forest_count)
        self.X=[pred_list,p_list,X[-self.timedelta:].values]
        return X_pred
    def one_step(self,X=None,actions=None):
        #X - это стартовое состояние мира. Один вектор
        if X is None:
            #внутреннее состояние - предполагаемая картина мира, какой-то старый прогноз
            X=self.X
        if type(X)==type(self.df_ethalon):
            #распотрошить датафрейм
            X=np.array(X.values[0],ndmin=2)
        if type(X)==type([]):
            histogram_input=True
        else:
            histogram_input=False
        #присобачить экшны       
        if actions is not None:
            if histogram_input:
                for i in range(len(X[0])):
                    for a in range(len(self.action_fields_where)):
                        X[0][i][:,self.action_fields_where[a]]=actions[a]
            else:
                for a in range(len(self.action_fields_where)):
                    X[:,self.action_fields_where[a]]=actions[a]
        if type(X)!=type(np.array([1])):
            self.X=self.forest.predict_next(X,return_hist=True)
        else:
            #если на входе массив, то сделаем его хотя бы матрицей
            self.X=self.forest.predict_next(np.array(X,ndmin=2),return_hist=True)
        return self.X
    def update_likelyhood(self,X_fact):
        #X_fact - фактический X, в смысле здоровый такой датафрейм или матрица
        #Нафига переть весь? Ради нормировки!
        if type(X_fact)==type(self.df_ethalon):
            #распотрошить датафрейм
            X_fact=X_fact.values
        for i in range(len(self.X[0])):
            Y_pred_lm_normed = self.X[0][i]/(self.means_X+1e-20)
            Y_normed = X_fact[-self.timedelta:,:]/(self.means_X+1e-20)
            err = (Y_pred_lm_normed - Y_normed)**2
            #Ну да, тупая константа. Она почти в 10 раз больше той другой константы. 
            #Теоретически это позволяет очень мощно учитывать последние наблюдения
            if np.sum(err)>self.forest.forest[i].log_likelyhood*1e7 and i==0:
                print('np.sum(err)>self.forest.forest[i].log_likelyhood*1e7',np.sum(err))
                #если у нас гипотезы настолько расходятся с фактами - значит, 
                #у нас ошибка в оценке гипотез. Просто забьём на это.
                break
                
            self.forest.forest[i].log_likelyhood+=0.02*np.sum(err)/err.shape[1]
        print('log_likelyhood ',[self.forest.forest[i].log_likelyhood for i in range(len(self.forest.forest))])
    def many_steps(self,X=None,actions=None,steps_count=20,return_reward=False,
                   reward_std_coef=0,disco_return=0.9,verbose=False,debug=False):
        #actions:  dict: action, step_number
        # [{ [10,-1,2],0 }, { [1,-1,2],2 }, { [1,-1,2],5 }  ]
        if X is None:
            #внутреннее состояние - предполагаемая картина мира, какой-то старый прогноз
            X = self.X
        if type(X)==type(self.df_ethalon):
            #распотрошить датафрейм
            X=X.values
        if type(X)==type([]):
            histogram_input=True
        else:
            histogram_input=False
            #X=[X,1,X]
            pred_list=[]
            p_list=[]
            for i in range(self.forest_count):
                pred_list.append(X)
                p_list.append(1./self.forest_count)
            X=[pred_list,p_list,X]
            
        j=0
        #средний реворд по всем траекториям. Все реворды дисконтированные
        reward_disco_avg=0.
        reward_disco_std=0.
        disco_cur=1
        forest_size=len(self.X[0])
        
        self.X=X
       
        if debug:
            l_hist=[]
            for i in range(forest_size):
                l_hist.append([])
                
        for i in range(steps_count):
            action_cur=None
            if len(actions)>j:
                if  actions[j]['step_number']==i:
                    action_cur=actions[j]['action']
                    j+=1
            self.one_step(self.X,action_cur)
               
            rew_cur=[]
            for k in range(forest_size):
                    
                p=self.X[1][k]
                X_cur=self.X[0][k][0]
                if verbose:
                    print('X_cur',X_cur)
                    print('X_cur[self.reward_field_where]*p*disco_cur',X_cur[self.reward_field_where],p,disco_cur)
                reward_disco_avg+=X_cur[self.reward_field_where]*p*disco_cur
                if p>(1e-4)*np.max(self.X[1][:]):
                    rew_cur.append(X_cur[self.reward_field_where])
                if debug:
                    l_hist[k].append(X_cur)
                    
            reward_disco_std+=np.std(rew_cur)*disco_cur
            disco_cur*=disco_return
        
        if debug:
            out_hist=[]
            out_hist.append(l_hist)
            out_hist.append([])
            for k in range(forest_size):
                l_hist[k]=np.vstack(l_hist[k])
                out_hist[1].append(self.X[1][k])
            return out_hist
        
        if return_reward:
            return reward_disco_avg+reward_std_coef*reward_disco_std
            
        
#ПРОТОКОЛ ГИСТОГРАММЫ: [pred_list,p_array,m_pred]


#какие переменные.
#пример X, action_fields, reward_field
#варианты действий
#на какую глубину прогнозировать, disco
#какие параметры эволюции в управлении
#какие параметры в эволюции в выборе моделей
#какие вообще параметры forest
class AIXI_controller(object):
    def __init__(self,X_example,action_fields,actions_set,actions_set_type_iter,
                 reward_field,planning_horizon,disco_rewards,plan_len,timedelta=1):
        #actions_set - это list из list-ов. Каждый list - это конкретный экшн.
        #Но если actions_set_type_iter=False - это сигнал, что надо интерпретировать как список возможных значений каждой команды
        self.sample_part=0.1
        self.forest_count=2
        self.boosting_count=2
        self.boosting_eta=0.5
        self.model_maxiter=2
        self.model_popsize=30
        self.model_mutation_p=0.1
        self.model_mutation_p_e=0.1
        self.model_mutation_r=0.2
        self.n_jobs=4
        self.model_alpha_count=4
        self.model_elitarism=2
        self.action_fields=action_fields
        self.reward_field=reward_field
        self.planning_horizon=planning_horizon
        self.disco_rewards=disco_rewards
        self.timedelta=timedelta
        
        if actions_set_type_iter:
            #То есть у нас тупо перечисление допустимых векторов
            self.actions_set=actions_set
        else:
            #То есть у нас тупо перечисление допустимых значений каждой переменной
            #а вектора строй сам
            self.actions_set=[list(elem_set) for elem_set in itertools.product(*actions_set)]
        print('self.actions_set',self.actions_set)
        self.X_example=X_example.copy()
        self.memory_size=1000
        self.tact_count=3
        self.zero_dup=70
        self.size_genom=2500
        self.regularization=0.6
        self.disco_model=0.9
        
        self.action_maxiter=3
        self.action_popsize=16
        self.action_mutation_p=0.1
        self.action_mutation_p_e=0.1
        self.action_mutation_r=0.2
        self.action_alpha_count=3
        self.action_elitarism=1
        #сколько действий может быть в плане, который в action
        self.plan_len=plan_len
        self.seed_action=5
        self.scouting_std=0.2
        self.scouting_random_prob=0.05
        self.scouting_predicted_prob=0.3
        self.timefield='tm'
    def fit(self,X,init_horizon=None,episodes_max_len=250,episodes_min_len=10,
            episodes_percent=0.8,seed=0,verbose=True):
        #init_horizon - горизонт инициализации
        self.realmod = reality_model(tact_count=self.tact_count,zero_dup=self.zero_dup,
                                     size_genom=self.size_genom,regularization=self.regularization,
                                     action_fields=self.action_fields,reward_field=self.reward_field,
                                     timedelta=self.timedelta, timefield=self.timefield,disco=self.disco_model)
        self.realmod.fit(X,seed=seed,episodes_min_len=episodes_min_len,episodes_max_len=episodes_max_len,
                         episodes_percent=episodes_percent,forest_count=self.forest_count,sample_part=self.sample_part,
                         boosting_count=self.boosting_count,
                         boosting_eta=self.boosting_eta,maxiter=self.model_maxiter,popsize=self.model_popsize,
                         mutation_p=self.model_mutation_p,mutation_p_e=self.model_mutation_p_e,
                         mutation_r=self.model_mutation_r,n_jobs=self.n_jobs,alpha_count=self.model_alpha_count,
                         elitarism=self.model_elitarism,verbose=verbose)
        if init_horizon is None:
            init_horizon=int(X.shape[0]/3)
        self.realmod.init_prediction(X,init_horizon)
    def draw(self,X,hist=False,recursive_foreсast=False,max_lst=250,what_to_forecast=0,plan=[]):
        #recursive_foresast - прогнозируем через many_steps
        #what_to_forecast - работает только для recursive forecast (нет)
        #plan - работает только если recursive forecast
        #horizon - работает только для  recursive forecast
        try:
            sym=copy.deepcopy(self.realmod.forest)
            if not recursive_foreсast:
                if hist:
                    [pred,p_array,m_pred]=np.array(sym.predict(X.values,return_hist=hist))
                else:
                    pred=np.array(sym.predict(X.values,return_hist=hist))

                figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                if what_to_forecast==0:
                    num=self.realmod.reward_field_where
                else:
                    num=what_to_forecast
                Y=X.shift(-1).values[:,num]
                plt.plot(Y[-max_lst:])
                if not hist:
                    plt.plot(pred.T[num,-max_lst:])
                else:
                    for i in range(len(pred)):
                        if 7+np.log(p_array[i])>0:
                            plt.plot(pred[i][-max_lst:,num],'--',linewidth=0.4*np.max([7+np.log(p_array[i]),0] ) )    
                plt.show()
            else:
                max_lst
                plan_parsed=[]
                #[{ [10,-1,2],0 }, { [1,-1,2],2 }, { [1,-1,2],5 }  ]
                chunk_size = 1+len(self.action_fields)
                plan = plan[:int(np.floor(len(plan)/chunk_size)*chunk_size)]
                for i in range(0,len(plan),chunk_size):
                    d={}
                    d['step_number']=plan[i]
                    d['action']=plan[i+1:i+chunk_size]
                    plan_parsed.append(d)
                #сделать виртуалку!
                reality_copy=copy.deepcopy(self.realmod)
                out_hist=reality_copy.many_steps(None,return_reward=False,steps_count=max_lst,
                                        verbose=False,actions=plan_parsed,disco_return=self.disco_rewards,
                                        reward_std_coef=self.scouting_std,debug=True)
                pred=out_hist[0]
                p_array=out_hist[1]
                for i in range(len(pred)):
                    if 7+np.log(p_array[i])>0:
                        plt.plot(pred[i][:,what_to_forecast],'--',linewidth=0.4*np.max([7+np.log(p_array[i]),0] ) )    
                plt.show()
        except Exception as e:
            print('couldnt draw',e)
    def estimate_plan(self,plan):
        plan_parsed=[]
        #[{ [10,-1,2],0 }, { [1,-1,2],2 }, { [1,-1,2],5 }  ]
        chunk_size = 1+len(self.action_fields)
        plan = plan[:int(np.floor(len(plan)/chunk_size)*chunk_size)]
        for i in range(0,len(plan),chunk_size):
            d={}
            d['step_number']=plan[i]
            d['action']=plan[i+1:i+chunk_size]
            plan_parsed.append(d)
        #сделать виртуалку!
        reality_copy=copy.deepcopy(self.realmod)
        rew=reality_copy.many_steps(None,return_reward=True,steps_count=self.planning_horizon,
                                    verbose=False,actions=plan_parsed,disco_return=self.disco_rewards,
                                    reward_std_coef=self.scouting_std)
        #костыль. Для совместимости с эволюцией
        return -rew
    def act(self,X,verbose=False):
        self.realmod.update_likelyhood(X)
        
        if np.random.rand()<self.scouting_random_prob:
            #делаем случайный ход
            action=self.actions_set[np.random.randint(0,len(self.actions_set))]
            if verbose:
                print('random action')
            self.realmod.one_step(actions=action)
            return action
        
        if np.random.rand()<self.scouting_predicted_prob:
            #делаем предсказанный ход
            if verbose:
                print('predicted action')
            #сделать виртуалку!
            reality_copy=copy.deepcopy(self.realmod)
            histo=reality_copy.one_step()
            X_pred=histo[2]
            action=[]
            for a in range(len(reality_copy.action_fields_where)):
                action.append(X_pred[:,reality_copy.action_fields_where[a]][0])
            action=np.array(action)
            self.realmod.one_step(actions=action)
            return action
            
            
        
        #записать X в self.X
        pred_list=[]
        p_list=[]
        for i in range(self.realmod.forest_count):
            pred_list.append(X[-1:].values)
            p_list.append(1./self.forest_count)
        self.realmod.X=[pred_list,p_list,X[-1:].values]
        
        
        #создать набор рандомных планов
        popsize=self.action_popsize
        maxiter=self.action_maxiter
        mutation_p=self.action_mutation_p
        mutation_p_e=self.action_mutation_p_e
        mutation_r=self.action_mutation_r
        alpha_count=self.action_alpha_count
        elitarism=self.action_elitarism
        seed=self.seed_action
        if seed>0:
            np.random.seed(seed)
        else:
            np.random.seed(np.random.randint(1e6))
        #сгенерить
        x_old = []
        prev_pointer=0
        for i in range(popsize):
            x=[]
            for j in range(self.plan_len):
                if j>0:
                    prev_pointer=np.random.randint(prev_pointer,high=self.planning_horizon)+0.
                x.append(prev_pointer)
                action_cur=self.actions_set[np.random.randint(0,len(self.actions_set))]
                x.extend(action_cur)
            x_old.append(np.array(x))
        for t in range(self.action_maxiter):
            pool = Pool(processes=self.n_jobs)
            l_out = pool.map(self.estimate_plan, [x for x in x_old])
            pool.close()
            pool.join()
            y_old = np.array(l_out)

            #отобрать альфачей
            alpha_nums = y_old.argsort()[:alpha_count]

            x_new = []
            for elit in range(elitarism):
                x_new.append(x_old[alpha_nums[elit]].copy()) 

            for child in range(popsize - elitarism):
                #скрещиваем
                crossed_alphas = alpha_nums[[np.random.randint(low=0,high=alpha_count),np.random.randint(low=0,high=alpha_count)]]
                x_c = x_old[alpha_nums[0]]
                idx = np.random.rand(len(x_c))<0.5
                x_c[idx] = x_old[alpha_nums[1]][idx]
                x_new.append(x_c)
                idx_muta = np.random.rand(len(x_c))<mutation_p
                x_c[idx_muta] += (np.random.rand(len(x_c[idx_muta]))-0.5)*2*mutation_r
                x_new.append(x_c.copy())

            x_old = x_new
            mutation_p = mutation_p*(1-mutation_p_e)

        pool = Pool(processes=self.n_jobs)
        l_out = pool.map(self.estimate_plan, [x for x in x_new])
        pool.close()
        pool.join()
        y_old = np.array(l_out)
        alpha_nums = y_old.argsort()[:alpha_count]
        
        final_plan=x_new[alpha_nums[np.argmin(y_old[alpha_nums])]]
        #print('final_plan',final_plan,'profit',-np.argmin(y_old[alpha_nums]))
        action=final_plan[1:1+len(self.action_fields)]
        self.realmod.one_step(actions=action)
        if verbose==2:
            print('final_plan',final_plan,'exp_reward',-min(y_old),'exp_rewards',-np.sort(y_old)[:4])
        if verbose==1:
            print('exp_reward',-min(y_old),'exp_rewards',-np.sort(y_old)[:4])
        return action
#ПРОТОКОЛ ГИСТОГРАММЫ: [pred_list,p_array,m_pred]       
