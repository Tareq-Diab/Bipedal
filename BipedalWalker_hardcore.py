


from keras import models , layers , optimizers
import numpy as np
import gym
import random
from collections import deque
from tqdm import tqdm
import pickle
import tensorflow as tf


#data normalization class 
class normalizer(object):
    def __init__(self,data):
        self.data=data
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        self.normalizer= MinMaxScaler()
        data=pd.read_csv(self.data)
        self.normalizer.fit(data.drop(["Unnamed: 0"],axis=1))
    def normalize(self,data):
        self.normalizer.transform()
#agent class
class DQN_AGENT:
    def __init__(self,state_space,action_space,reply_memory,epsilon=1,epsilon_min=0.01,epsilon_decay=0.99,learning_rate=0.005,discount_rate=0.99,discretization=True,discretization_resolution=2):
        self.action_space=action_space
        self.state_space=state_space
        self.reply_memory_size=reply_memory
        self.gamma = discount_rate
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        self.learning_rate=learning_rate
        self.memory=self.build_memory()
        self.model=self.build_model()
        self.discretization=discretization


    def build_memory(self):
        return deque(maxlen=self.reply_memory_size)

    def add_to_memory(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def build_model(self):
        model=models.Sequential()
        model.add(layers.Dense(1024,input_shape=(self.state_space,),activation='relu'))
        model.add(layers.Dense(1024,activation='relu'))
        model.add(layers.Dense(self.action_space))
        model.compile(loss='mse',optimizer=optimizers.Adam(lr=self.learning_rate))
        return model



    def take_action(self,state,trainig=True):

        """
        i picked 6 action in the following representation 
        -action 1 and 2 controll the main engine where  -1..0 off, 0..+1 throttle from 50% to 100% power
        assighning action_1 to 0 (engine off) action_2 to 1 (engine on full throttle) 
        - action_3 & action_4 controlls right engine , where action_3 is +0.5 (50% throttle) and action_4 is +1 (100% throttle)
        - action_5 & action_6 controlls left engine  , where action_5 is -0.5 (50% throttle) and action_6 is -1 (100% throttle)
        """
        
        if trainig:

            if np.random.rand() <= self.epsilon:
                d=[-1. , -0.7, -0.5, -0.3,   0. ,  0.3,   0.5 , 0.7,  1.]
                act=np.copy([random.sample(d,1)[0] , random.sample(d,1)[0] , \
                            random.sample(d,1)[0]  , random.sample(d,1)[0] ]).astype(float)
                #d={2:0.5,3:1,4:-0.5,5:-1}
                #act[1]=d[act[1]]
                return act
                """
                selecting [action for main motor,action for choosing between left and right motor]
                """
            else:
                q_action=self.model.predict(state.reshape([1,env.observation_space.shape[0]]))
                n=9
                act=np.copy([np.argmax(q_action[0][0:n]),np.argmax(q_action[0][n:n*2]),np.argmax(q_action[0][n*2:n*3]),np.argmax(q_action[0][n*3:n*4])]).astype(float)
                #mapping index to discret action values
                d=[-1. , -0.7, -0.5, -0.3,   0. ,  0.3,   0.5 , 0.7,  1.]
                for i in range(4):
                    act[i]=d[int(act[i])]
                #print("__________________________________________________________________")
                #print(q_action)
                #print(act)
                #print("__________________________________________________________________")
                return act

                """
                selecting [action for main motor,action for choosing between left and right motor]
                """
        q_action=self.model.predict(state)
        return np.argmax(q_action[0])



    def load(self,name):
        self.model.load_weights(name)

        
    def save(self,name):
        self.model.save_weights(name)
    
    def action_to_index(self,action):
        d=[-1. , -0.7, -0.5, -0.3,   0. ,  0.3,   0.5 , 0.7,  1.]
        n=9
        """
        d0={-1:0,-0.5:1,0:2,0.5:3,1:4}
        d1={-1:5,-0.5:6,0:7,0.5:8,1:9}
        d2={-1:10,-0.5:11,0:12,0.5:13,1:14}
        d3={-1:15,-0.5:16,0:17,0.5:18,1:19}
        """

        return [d.index(action[0]),n+d.index(action[1]),2*n+d.index(action[2]),3*n+d.index(action[3])]

    def train(self,batch_size):
        n=9
        minibatch=np.array(random.sample(self.memory,batch_size))
        s,a,r,n_s,done = minibatch[:,0],minibatch[:,1],minibatch[:,2],minibatch[:,3],minibatch[:,4]

        future_q=np.array([[ np.amax(q[0:n]) , np.amax(q[n:n*2]) ,np.amax(q[n*2:n*3]), np.amax(q[n*3:n*4])  ] for q in self.model.predict(np.stack(n_s))])

        target=r.reshape([-1,1])+self.gamma*future_q*(1-done.reshape([-1,1]))

        target_f=self.model.predict(np.stack(s))

        for t_f,action,t in zip(target_f,np.stack(a),target):
            #print("=====================================================")
            #print(t_f)
            #print(t)
            #print(action)
            action=self.action_to_index(action)
            #print(action)
            t_f[int(action[0])]=t[0]
            t_f[int(action[1])]=t[1]
            t_f[int(action[2])]=t[2]
            t_f[int(action[3])]=t[3]
            #print(t_f)
            ##print("=====================================================")

        self.model.fit(np.stack(s),target_f,epochs=1,verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay




env=gym.make('BipedalWalkerHardcore-v3')
def trymodel(r_train,episode):
    result=[]
    env=gym.make('BipedalWalkerHardcore-v3')
    agent_test=DQN_AGENT(env.observation_space.shape[0],36,reply_memory=0,epsilon=0)# 6 is the number of the dicrite action i chose 
    #agent_test.load("rlmodels/BipedalWalker-hardcore/model_score_{} , episode_{}".format(276,331))

    agent_test.load("rlmodels/BipedalWalker-hardcore/model_score_{} , episode_{}".format(r_train,episode))
    for i in range(15):
        s=env.reset()
        done =False
        r_train=[]
        steps=0
        while not done :
            env.render()
            a=agent_test.take_action(s)
            n_s,r,done,_=env.step(a)
            s=n_s
            r_train.append(r)
            if done:
                result.append(np.sum(r_train))

                print("r {} ".format( np.sum(r_train)))
    env.close()
    return np.sum(result)/15
                
def test(r_train,episode):
    result=[]
    env=gym.make('BipedalWalkerHardcore-v3')
    agent_test=DQN_AGENT(env.observation_space.shape[0],36,reply_memory=0,epsilon=0)# 6 is the number of the dicrite action i chose 
    #agent_test.load("rlmodels/BipedalWalker-hardcore/model_score_{} , episode_{}".format(276,331))

    agent_test.load("rlmodels/BipedalWalker-hardcore/model_score_{} , episode_{}".format(int(np.sum(r_train[episode])),episode))
    for i in range(15):
        s=env.reset()
        done =False
        r_train=[]
        steps=0
        while not done :
            env.render()
            a=agent_test.take_action(s)
            n_s,r,done,_=env.step(a)
            s=n_s
            r_train.append(r)
            if done:
                result.append(np.sum(r_train))

                print("r {} ".format( np.sum(r_train)))
    env.close()
    return np.sum(result)/15
                




if __name__ == "__main__":
    with tf.device("gpu:0"):
        #trymodel(285,596) the res 6 normal

        r_train=[]
        log=[]
        stop_training=False
        env=gym.make('BipedalWalkerHardcore-v3') 
        agent=DQN_AGENT(env.observation_space.shape[0],36,reply_memory=100000,learning_rate=0.0001,epsilon_decay=0.99,epsilon=1)# 6 is the number of the dicrite action i chose


        done =False
        batch_size=64
        for episode in tqdm(range(7000)):

            done =False
            steps=0
            s=env.reset()

            er=[]
            #---------------------------------------------------Testing-------------------------------------------------------
            if stop_training :
                x=test(r_train,(episode-1))
                print("test result {}".format(x))
                if x > 200:
                    env.close()
                    print("stopped trainiing after score {}".format(np.sum(r_train[episode-1])))
                    break 

                stop_training = False
                print("exiting test and turning stop_training back to {}".format(stop_training))
            #---------------------------------------------------Testing-------------------------------------------------------

            while not done :

                if episode % 20 ==0:
                    env.render()

                a=agent.take_action(s)
                n_s,r,done,_=env.step(a)
                agent.add_to_memory(s,a,r,n_s,done)
                s=n_s
                er.append(r)
                if done:


                    r_train.append(er)
                    print("episode: {} / {}, score: {}, e{}".format(episode, 1000, np.sum(r_train[episode]),agent.epsilon))
                    log.append([episode,r_train[episode]])
                    if np.sum(r_train[episode]) > 100:
                        agent.save("rlmodels/BipedalWalker-hardcore/model_score_{} , episode_{}".format(int(np.sum(r_train[episode])),episode))
                    if np.sum(r_train[episode]) > 200 :
                        stop_training =True
                        print("stopping training because reward = {} at episode {}".format(np.sum(r_train[episode]),episode))
                    break 

                if len(agent.memory) > batch_size:
                    agent.train(batch_size)

        outfile=open("rlmodels/BipedalWalker-hardcore/log",'wb')
        memory=open("rlmodels/BipedalWalker-hardcore/memory",'wb')

        pickle.dump(log,outfile)
        pickle.dump(agent.memory,memory)
        outfile.close()





