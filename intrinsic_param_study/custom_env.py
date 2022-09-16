import numpy as np, pandas as pd, os, gym, shutil, copy
from pandas import DataFrame as DF
import pandapower.networks as pn, pandapower as pp
from pandapower import ppException
import matplotlib.pyplot as plt
from gym.utils import seeding
from gym import spaces

class GridEnv(gym.Env):
    params = {
        'percent_flexible': 1, 	# percentage of one
        'v_margin': 0.07, 		# voltage margin percentage of one
        'i_max': 90,
        'eval_path': 'evaluation',
        'snr_db': 30,
        'demand_scale_factor': 500,	# n_bus=500 | two_bus=200000
        'gen_scale_factor': 100,	# n_bus=100 | two_bus=10000
        'use_agent': True,
        'dtype': np.float32
    }
           
    def __init__(self, powergrid, total_timesteps, model=None, evaluation=False, seed=0, id_=0,
                 T=24, tau=24, beta_1=1, beta_2=1, beta_3=1, beta_4=1, beta_5=1, beta_6=1):
        super(GridEnv, self).__init__()
        
        self.custom_hyper = {}
        self.custom_hyper['T'] = T
        self.custom_hyper['tau'] = tau
        self.custom_hyper['beta_1'] = beta_1
        self.custom_hyper['beta_2'] = beta_2
        self.custom_hyper['beta_3'] = beta_3
        self.custom_hyper['beta_4'] = beta_4
        self.custom_hyper['beta_5'] = beta_5
        self.custom_hyper['beta_6'] = beta_6
        
        # In case divergence starts from the beginning.
        self.v_violations = np.nan
        self.i_violations = np.nan
        self.losses = np.nan
        
        self.total_timesteps = total_timesteps
        self.model = model
        self.evaluation = evaluation
        self.id_ = id_
        self.seed = seed
        
        # Evaluation folder
        if self.evaluation:
            path = os.path.join(os.getcwd(), self.params['eval_path'])
            os.makedirs(path, exist_ok=True)
            self.path = path
        
        # POWER GRID        
        self.init_grid = powergrid
        pp.runpp(self.init_grid) # compute power flow
        self.grid = copy.deepcopy(self.init_grid)
        
        # DATASET
        self.dataset = data_augmentation('dataset.csv', self.evaluation, total_timesteps,
                                         self.seed, self.params['snr_db'])
                                         
        #if self.evaluation: self.demand = self.dataset.loc[:, ['demand_forecast']]
        #else: self.demand = self.dataset.loc[:, ['demand']]
        
        self.demand = self.dataset.loc[:, ['demand']]	#DEMAND
        self.gen = self.dataset.loc[:, ['gen_solar']]	#GENERATION
        self.ir = self.dataset.loc[:, ['ir']]			#IRRADIATION
        
        # SCALING
        self.scale_demand = self.grid.load['p_mw']/self.grid.load['p_mw'].sum() #eq. 4.5.4
        self.scale_demand /= self.params['demand_scale_factor']
        self.scale_gen = self.grid.sgen['p_mw']/self.grid.sgen['p_mw'].sum() #eq. 4.5.5
        self.scale_gen /= self.params['gen_scale_factor']
        
        # CUSTOM SPACE
        
        self.n_flexible_loads = int(np.ceil(len(self.grid.load)*self.params['percent_flexible']))
        self.action_old = np.zeros(self.n_flexible_loads)
        self.action_space = spaces.Box(0., 1., shape=(self.n_flexible_loads,),
                                       dtype=self.params['dtype'])
        
        # for REPRODUCIBILITY porpuses
        self.action_space.seed(self.seed)
        
        # ENV INSTANCES        
        self.reward = 0
        self.done = False
        self.t = 0
        self.QPratio = self.grid.load['q_mvar']/self.grid.load['p_mw'] #=tan(phi), FP = cos(phi)
        self.load_modulation = DF(np.zeros((len(self.grid.load), self.custom_hyper['tau'])))
        
        # OBSERVATION
        
        # Observation subsets (T previous states + actual state)
        self.loads_p = DF(np.zeros((len(self.grid.res_load['p_mw']), self.custom_hyper['T']+1)))
        self.sgen_p = DF(np.zeros((len(self.grid.res_sgen['p_mw']), self.custom_hyper['T']+1)))
        self.gen_p = DF(np.zeros((len(self.grid.res_gen['p_mw']), self.custom_hyper['T']+1)))
        self.gen_q = DF(np.zeros((len(self.grid.res_gen['q_mvar']), self.custom_hyper['T']+1)))
        self.ext_p = DF(np.zeros((1, self.custom_hyper['T']+1)))
        self.ext_q = DF(np.zeros((1, self.custom_hyper['T']+1)))
        self.ir_ = DF(np.zeros((1, self.custom_hyper['T']+1)))
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape = self.get_observation().shape,
                                            dtype=self.params['dtype'])
    
    def save_metrics(self): #for later analysis
    
        if self.model == None: path = os.path.join(self.path, f"{self.id_}_{self.seed}"+'.csv')
        else: path = os.path.join(self.path, f"{self.model}_{self.id_}_{self.seed}"+'.csv')
        
        df = DF({'reward':[], 'v_violations':[], 'i_violations':[], 'losses':[], 'demand':[]})
        
        df.loc[self.t, 'reward'] = self.reward
        df.loc[self.t, 'v_violations'] = self.v_violations
        df.loc[self.t, 'i_violations'] = self.i_violations
        df.loc[self.t, 'losses'] = self.grid.res_line['pl_mw'].sum()
        # Agent demand rescaled
        df.loc[self.t, 'demand'] = self.grid.res_load['p_mw'].sum() * \
                                   self.params['demand_scale_factor']
        
        hyper_set = DF.from_dict([self.custom_hyper])
        hyper_set.index = [self.t]
        
        df = pd.concat([df, hyper_set], axis=1)        
        df.to_csv(path, mode='a', header = not os.path.exists(path))
        
    def load_modulation_curve(self, action):
        
        # put zeros where there is no flexible loads
        act = np.append(action, np.zeros(len(self.grid.load['p_mw'])-self.n_flexible_loads))
        
        self.load_modulation.iloc[:, :-1] = self.load_modulation.iloc[:, 1:]
        self.load_modulation.iloc[:, -1] = act*(self.loads_p.iloc[:,-1] - self.grid.load['p_mw'])
        
        return self.load_modulation
    
    def step(self, action):
        
        #TRANSITION FUNCTION    
        # UPDATE DEMAND eq. 4.5.6
        self.grid.load['p_mw'] = self.demand.iloc[self.t, 0] * self.scale_demand
        
        # UPDATE PRODUCTION eq. 4.5.9
        self.grid.sgen['p_mw'] = self.gen.iloc[self.t, 0] * self.scale_gen
        
        if self.done == False:
            
            # ACTIONS
            if self.params['use_agent'] == False:
                # actions will always be 0 (no actions)
                action = np.zeros(self.n_flexible_loads)
            
            self.Delta_P = self.load_modulation_curve(action).iloc[:, -1] #eq 4.4.1
            self.grid.load['p_mw'] += self.Delta_P # eq. 4.5.7
            self.grid.load['q_mvar'] = self.grid.load['p_mw'] * self.QPratio
            
            # Solve power flow and update system
            try:
                pp.runpp(self.grid)
                self.reward = self.get_reward(action)
                self.done = False # Converges, it is not done.
                self.observation = self.get_observation()
                if self.evaluation:
                    self.save_metrics()

            except ppException as err:
                print("-> POWER FLOW COULD NOT CONVERGE:", err)
                
                # penalise for divergence
                self.reward = -100*self.custom_hyper['beta_6']
                self.done = True
                self.observation = self.reset()
        
        if self.t >= self.total_timesteps -1: self.done = True
        self.action_old = action
        
        # END TRANSITION FUNCTION
        info = {
            'real_demand': self.dataset.loc[:, ['demand']].iloc[self.t, 0],
            'result': self.grid.res_load['p_mw'].sum() * self.params['demand_scale_factor'], 
            'load_modulation': self.Delta_P.sum()*self.params['demand_scale_factor'],
            'v_violations': self.v_violations,
            'i_violations': self.i_violations,
            'losses': self.losses
        }
        self.t += 1
        
        return self.observation, self.reward, self.done, info
        
    def get_reward(self, action):
    
        #EL REWARD
        reward = 0
        
        # r¹        
        v = self.grid.res_bus['vm_pu']
        v_min = 1 - self.params['v_margin']
        v_max = 1 + self.params['v_margin']
        v_lower = sum(v_min - v[v < v_min]) 
        v_upper = sum(v[v > v_max] - v_max) #v[v < v_min] & v[v > v_max] are violations
        self.v_violations = v_lower + v_upper
        reward -= self.v_violations*self.custom_hyper['beta_1']
        
        # r²        
        i = self.grid.res_line['loading_percent'] # congestion percentage
        i_max = self.params['i_max']
        self.i_violations = sum(i[i > i_max] - i_max)/100
        reward -= self.i_violations*self.custom_hyper['beta_2']
        
        # r³
        activation_cost = self.action_old.sum() #how previous actions affect the current reward
        reward += 10*self.n_flexible_loads*activation_cost*self.custom_hyper['beta_3']
        
        # r⁴        
        self.losses = self.grid.res_line['pl_mw'].sum()
        reward -= self.losses*self.custom_hyper['beta_4']
        
        # r⁵
        if (self.t + 1) % self.custom_hyper['tau'] == 0:
            reward -= abs(self.load_modulation.sum(axis=1).sum())/self.n_flexible_loads*\
                          self.custom_hyper['beta_5']
        
        return reward

        
    def get_observation(self):
        
        # empty lists can be added, in case a type of generation does not exist in the grid.
        observation = []
                
        # S = S¹ x S² x S³
        # S¹ + S³ (analogous)
        self.loads_p = pd.concat([self.loads_p.iloc[:, 1:], self.grid.res_load['p_mw']], axis=1)
        # grid.res_load is the resulting modulated demand
        observation += list(self.loads_p.values.ravel())
        self.ir_ = pd.concat([self.ir_.iloc[:,1:],DF(np.array([self.ir.iloc[self.t,0]]))],axis=1)
        #observation += list(self.ir_.values.ravel())
        
        # S² + S³ (analogous)
        self.sgen_p = pd.concat([self.sgen_p.iloc[:, 1:], self.grid.res_sgen['p_mw']], axis=1)
        observation += list(self.sgen_p.values.ravel())
        self.gen_p = pd.concat([self.gen_p.iloc[:, 1:], self.grid.res_gen['p_mw']], axis=1)
        observation += list(self.gen_p.values.ravel())
        self.ext_p = pd.concat([self.ext_p.iloc[:, 1:], self.grid.res_ext_grid['p_mw']], axis=1)
        observation += list(self.ext_p.values.ravel())
        
        self.gen_q = pd.concat([self.gen_q.iloc[:, 1:], self.grid.res_gen['q_mvar']], axis=1)
        observation += list(self.gen_q.values.ravel())
        self.ext_q = pd.concat([self.ext_q.iloc[:, 1:], self.grid.res_ext_grid['q_mvar']],axis=1)
        observation += list(self.ext_q.values.ravel())
                        
        return np.array(observation).astype(self.params['dtype'])

    def reset(self):
        
        self.grid = copy.deepcopy(self.init_grid)
        self.reward = 0
        self.done = False
        
        self.load_modulation = DF(np.zeros((len(self.grid.load), self.custom_hyper['tau'])))
        
        # Observation subsets (T previous states + actual state)
        self.loads_p = DF(np.zeros((len(self.grid.res_load['p_mw']), self.custom_hyper['T']+1)))
        self.sgen_p = DF(np.zeros((len(self.grid.res_sgen['p_mw']), self.custom_hyper['T']+1)))
        self.gen_p = DF(np.zeros((len(self.grid.res_gen['p_mw']), self.custom_hyper['T']+1)))
        self.gen_q = DF(np.zeros((len(self.grid.res_gen['q_mvar']), self.custom_hyper['T']+1)))
        self.ext_p = DF(np.zeros((1, self.custom_hyper['T']+1)))
        self.ext_q = DF(np.zeros((1, self.custom_hyper['T']+1)))
        self.ir_ = DF(np.zeros((1, self.custom_hyper['T']+1)))
        
        return self.get_observation()  # returns obs
    
    def render(self):
        pass
    
    def close (self):
        pass
        
def data_augmentation(path_file, eval_, total_timesteps=365*24, seed=0, snr_db=30):    
    df = pd.read_csv(path_file)
    df.index = pd.to_datetime(df.iloc[:, 0], utc=True) + pd.DateOffset(hours=1)
    df.index.name = 'Timestamp'
    df = df.iloc[:, 1:]
    
    if eval_ == False:
        # Data duplication
        n_period = total_timesteps//len(df)+1
        idx = pd.date_range(start=df.index[0], periods=n_period*len(df), freq='1h')
        df_original = copy.deepcopy(df)
        for _ in range(n_period-1):
            df = df.append(df_original, ignore_index = True)
        df.index = idx #index correction
    
        # Adding SNR Gaussian white noise
        # snr_db => SNR en dB = 10log(Magnitud)
        # as more dB less noise.
        for i in range(len(df.columns)-1):
            sp_avg = np.mean(df.iloc[:, i+1]**2) #Signal Power average
            sp_dB_avg = 10*np.log10(sp_avg)
            noise_dB = sp_dB_avg - snr_db
            noise = 10**(noise_dB/10) # Magnitud = 10**(dB/10)
            # White Gaussian noise
            mu = 0
            std = np.sqrt(noise)
            noise = np.random.RandomState(seed).normal(mu, std, df.shape[0])
            df.iloc[:, i+1] += noise
            
    return df
