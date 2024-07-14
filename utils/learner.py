from collections import deque
from tqdm import tqdm
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
from mlagents_envs.environment import ActionTuple
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel # infos : https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-LLAPI.md

def to_uint8_grayscale(img):
    img = np.array(127.5*(img.cpu().detach()+1)).astype(np.uint8)
    return(img)

def to_tensor_grayscale(obs):
    return((2*(torch.tensor(obs).squeeze(0) - 0.5)).mean(axis=0))


class Segmenter:
    def __init__(self,ground=False):
        # Do to Unity compression (for fast running) we must threshold semantic maps to clean them.
        self.ground = ground
        self.ground_rgb = torch.tensor([0,0,0]).view(3,1,1)
        self.obstacle_rgb = torch.tensor([1,0,0]).view(3,1,1)
        self.wall_rgb = torch.tensor([0,1,0]).view(3,1,1)
        self.coin_rgb = torch.tensor([0,0,1]).view(3,1,1)
    def segment(self,seg):
        tmp = torch.zeros(128,128)
        wall = ((seg-self.wall_rgb).abs().mean(axis=1)<0.05)[0]
        coin = ((seg-self.coin_rgb).abs().mean(axis=1)<0.05)[0]
        obstacle = ((seg-self.obstacle_rgb).abs().mean(axis=1)<0.05)[0]
        if self.ground:
            ground = ((seg-self.ground_rgb).abs().mean(axis=1)<0.05)[0]
            tmp[ground] = 0.5
        tmp[coin] = 1
        tmp[wall] = -0.5
        tmp[obstacle] = -1
        
        
        return(tmp)
    def to_uint8(self,result):
        return(np.array((127.5*(result+1))).astype(np.uint8))


class FrameReplayBuffer:
    def __init__(self, buffer_size, device):
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size) 

    def add_experience(self, state, next_state, seg, action, reward, action_state,next_action_states):
        self.buffer.append((state, next_state, seg,  action, reward, action_state,next_action_states))


    
    def sample_batch(self, batch_size):

        idx = np.random.randint(0, self.buffer.__len__(),batch_size)

        batch_unprocessed = [self.buffer[i] for i in idx]
        
        
        batch_states = torch.stack([b[0] for b in batch_unprocessed]).to(self.device)
        batch_next_states = torch.stack([b[1] for b in batch_unprocessed]).to(self.device)
        batch_seg = torch.stack([b[2] for b in batch_unprocessed]).to(self.device)
        batch_action_states = torch.stack([b[5] for b in batch_unprocessed]).to(self.device)
        batch_next_action_states = torch.stack([b[6] for b in batch_unprocessed]).to(self.device)
        batch_actions = torch.tensor([b[3] for b in batch_unprocessed]).unsqueeze(1).to(self.device)
        batch_rewards = torch.tensor([b[4] for b in batch_unprocessed]).unsqueeze(1).to(self.device)



        return(batch_states,batch_next_states,batch_seg,batch_actions,batch_rewards,batch_action_states,batch_next_action_states)


    def __len__(self):
        return len(self.buffer)


class RLLearner:
    def __init__(self, 
                 env, 
                 model,
                 target_model, 
                 learning_rate = 0.01,
                 buffer_size=5000, 
                 gamma = 0.99,
                 batch_size=64,
                 eval_every = 300,
                 update_every = 1000,
                 epsilon_decrease = 500,
                 episode_duration = 200,
                 action_mapper = None,
                 train_every = 10,
                 device = "cpu",
                 segmenter = Segmenter(),
                 frame_size = 128,
                 do_clip = True,
                 epsilon_min = 0.05,
                 action_memory = 5,
                update_type = "hard",
                 n_frames = 3,
                tau = 0.01,
                 budget = 100000,
                env_settings = {"coin_proba":1, #parameters for the environments. 
                 "increase_obstacle_proba":1, #Linear increase in obstacle proportion until max_obstacle_proba is reached
                 "move_speed":[1,1], #Movement speed
                 "turn_speed":[150,150], #Rotation speed
                 "momentum":[0,0], #Inertial momentum
                 "decrease_reward_on_stay":0, #decrease reward when OnStayCollided method is called
                 "coin_visible":1, #Coins visibility
                 "max_obstacle_proba":0.3}, #Obstacle proportion
                using_seg_input=False,
                channel_env = None,
                eval_n = 10): #not starting training until  x steps
        self.segmenter = segmenter
        self.budget = budget
        self.do_clip = do_clip
        self.channel_env = channel_env
        self.frame_size = frame_size
        self.n_frames = n_frames
        self.env = env  # The RL environment
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.model = model.to(self.device)  # The RL model (e.g., a neural network)
        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.eval_every = eval_every
        self.update_every = update_every
        self.update_type = update_type
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.RP = FrameReplayBuffer(buffer_size,self.device)
        self.epsilon = 1
        self.epsilon_start = self.epsilon
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_min = epsilon_min
        self.total_timesteps = 0
        self.total_updates = 0
        self.episode = 0
        self.episode_duration = episode_duration
        self.rl_loss = nn.MSELoss()
        self.seg_loss = nn.MSELoss()
        self.action_mapper = action_mapper
        self.n_actions = len(self.action_mapper)
        self.train_every = train_every        
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.rewards_eval = []
        self.action_memory = action_memory
        self.tau = tau
        self.using_seg_input = using_seg_input
        self.max_grad = []
        self.env_settings = env_settings
        self.max_obstacle_proba = self.env_settings['max_obstacle_proba']
        self.eval_n = eval_n
        if self.env_settings['increase_obstacle_proba']:
            self.n_episodes = self.budget//self.episode_duration
            self.probas = np.ones(self.n_episodes+1)*self.max_obstacle_proba
            start_probas =self.max_obstacle_proba*(np.arange((self.budget//self.episode_duration)//2)/((self.budget//self.episode_duration)//2))
            self.probas[:start_probas.shape[0]] = start_probas


    def init_state(self):
        frame,_ = self.extract_frames()
        self.state = []
        for i in range(self.n_frames):
            self.state.append(frame)
        self.action_states = []
        for i in range(self.action_memory):
            self.action_states.append(torch.zeros(self.n_actions))
        #self.state_seg = seg
        
    def extract_frames(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        reward = decision_steps.reward
        segmented_o, rgb_o = decision_steps.obs
        
        
        if self.using_seg_input:
            frame = self.segmenter.segment(torch.tensor(segmented_o))
        else:
            frame = to_tensor_grayscale(rgb_o)
        return(frame,reward)

    def update_state(self,action_n = None): #used to update current state wth 3 latest frame and current segmentation map
        frame, _ = self.extract_frames()
        self.state.pop(0)
        self.state.append(frame)
        #self.state_seg = seg
        self.action_states.pop(0)
        action_tensor = torch.zeros(self.n_actions)
        if action_n is not None:
            action_tensor[action_n] = 1
        self.action_states.append(action_tensor)
        
    def extract_reward(self):
        return(self.env.get_steps(self.behavior_name)[0].reward)
        
    def select_random_action(self):
        action_tuple = ActionTuple()
        action_n = np.random.randint(self.n_actions) #N ACTIONS POSSIBLE
        return(action_n)
        
    def select_action(self):
        if np.random.uniform()<self.epsilon:
            action_n = self.select_random_action()
            return(action_n)
        else:
            input = torch.stack(self.state).unsqueeze(0).to(self.device)
            input_action = torch.stack(self.action_states).unsqueeze(0).to(self.device)
            seg, values = self.model(input,input_action)
            action_n = torch.argmax(values[0]).item()
            return(action_n)
            
    def select_best_action(self):
        input = torch.stack(self.state).unsqueeze(0).to(self.device)
        input_action = torch.stack(self.action_states).unsqueeze(0).to(self.device)
        seg, values = self.model(x=input,y=input_action,eval=False) #to change ?
        action_n = torch.argmax(values[0]).item()
        return(action_n)
        
            
    def generate_action_tuple(self,action_n):
        forward, turn = self.action_mapper[action_n]
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[forward,turn]])) #[forwardn turn] where turn is -1, 0 or 1
        return(action_tuple)
        
    def reset_environment(self):
        self.env.reset()


    def new_episode(self,eval = False):

        
        move_speed = np.random.uniform(self.env_settings["move_speed"][0],self.env_settings["move_speed"][1]) if not eval else 1
        momentum = np.random.uniform(self.env_settings["momentum"][0],self.env_settings["momentum"][1]) if not eval else 0
        turn_speed = np.random.uniform(self.env_settings["turn_speed"][0],self.env_settings["turn_speed"][1]) if not eval else 150
        coin_proba = self.env_settings["coin_proba"] if not eval else 1
        decrease_reward_on_stay = self.env_settings["decrease_reward_on_stay"] if not eval else 0
        obstacle_proba = (self.max_obstacle_proba if not self.env_settings["increase_obstacle_proba"] else self.probas[self.episode]) if not eval else self.max_obstacle_proba
        coin_visible = self.env_settings["coin_visible"]
        
        self.channel_env.set_float_parameter("coin_proba", coin_proba)
        self.channel_env.set_float_parameter("obstacle_proba",obstacle_proba)
        
        self.channel_env.set_float_parameter("move_speed",move_speed)
        self.channel_env.set_float_parameter("turn_speed",turn_speed)
        self.channel_env.set_float_parameter("momentum",momentum)
        self.channel_env.set_float_parameter("decrease_reward_on_stay",decrease_reward_on_stay)
        self.channel_env.set_float_parameter("coin_visible",coin_visible)

        
        self.env.reset()
        self.init_state()
        
    def clip(self, size,eval_mode = True):
        self.new_episode(eval_mode)
        frames = []
        action_n = None
        for i in range(size):
            self.update_state(action_n)
            frame = self.state[-1]
            
            action_n = self.select_best_action()
            #print(action_n)
            action = self.generate_action_tuple(action_n)

            #seg = self.state_seg
            self.env.set_actions(self.behavior_name, action)
            self.env.step()
            frames.append(to_uint8_grayscale(frame))
        self.new_episode(False)
        return(frames)
    def calculate_max_grad_norm(self):
        max_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                max_norm = max(max_norm, param_norm.item())
        return max_norm
    def eval(self,n_episodes,len_episode):
        #print('Starting Evaluation.')
        rewards_eval = 0

        for j in range(n_episodes):
            self.new_episode(True)
            reward_ep = 0
            action_n = None
            for i in range(len_episode):
                self.update_state(action_n)
                with torch.no_grad():
                    action_n = self.select_best_action()
                #print(action_n)
                action = self.generate_action_tuple(action_n)
                #seg = self.state_seg
                self.env.set_actions(self.behavior_name, action)
                self.env.step()
                reward_ep+= self.extract_reward().item()
            rewards_eval+=reward_ep
        self.new_episode(False)
        return(rewards_eval/n_episodes)

    def train_model(self):

        pbar = tqdm(range(self.budget))
        for i in pbar:
            if (self.total_timesteps % self.episode_duration == 0 ):
                if self.episode>=1:
                    
                    if self.do_clip:
                        print("Saving video...")
                        frames = self.clip(self.do_clip)
                        video = cv2.VideoWriter('video_episode_'+str(self.episode)+'.avi',cv2.VideoWriter_fourcc(*'XVID'),15, (self.frame_size,self.frame_size))
                        for image in tqdm(frames):
                            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        video.release()
                    R = self.eval(self.eval_n,400)
                    self.rewards_eval.append(R)
                    print('Mean reward result after episode ' + str(self.episode) + ' : ', R)
                self.episode+=1
                self.new_episode()
                


                
            self.total_timesteps +=1
            #update self.state
            
            current_state = self.state
            
            action_n = self.select_action()
            action = self.generate_action_tuple(action_n)

            #seg = self.state_seg
            seg = torch.tensor([0])
            self.env.set_actions(self.behavior_name, action)
            self.env.step()
            reward = self.extract_reward()
            if reward>0:
                reward-= 2
            old_state = self.state.copy()
            old_action_state = self.action_states.copy()
            self.update_state(action_n)
            self.RP.add_experience(torch.stack(old_state), torch.stack(self.state), seg, action_n, reward.item(),torch.stack(old_action_state), torch.stack(self.action_states))
            

            #if action_n==0:
            #    reward+=0.01
            if self.epsilon > self.epsilon_min:
                    self.epsilon -= (self.epsilon_start - self.epsilon_min) / self.epsilon_decrease


                
          # evaluate agent
            if ((self.RP.__len__()>=self.batch_size) and (self.total_timesteps % self.train_every == 0)):#check if needed condition isnt >=self.batch_size+3
                #print(self.select_best_action())
                batch_states,batch_next_states,batch_seg,batch_actions,batch_rewards, batch_action_states, batch_next_action_states = self.RP.sample_batch(self.batch_size)



                
                seg_network, q_values = self.model(x = batch_states,y = batch_action_states)
                values = q_values.gather(1,batch_actions)
                with torch.no_grad():
                    target = self.target_model(x = batch_next_states, y = batch_next_action_states)[1]
                    maxs, _ = target.max(axis=1)
                    maxs = maxs.unsqueeze(1)
                    target = batch_rewards + self.gamma*maxs
                
                rl_loss = self.rl_loss(values, target)
                loss = rl_loss
                #seg_loss = torch.tensor([0.])

                self.optimizer.zero_grad()
                loss.backward()
                self.max_grad.append(self.calculate_max_grad_norm())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                text = 'rl_loss : ' + str(rl_loss.item())
                pbar.set_description(text, refresh = True)
                #print('loss', loss.item(), ' / seg_loss', seg_loss.item(), ' / rl_loss', rl_loss.item(), ' / Timestep : ', self.total_timesteps)
                self.total_updates+=1
                del batch_states
                del batch_next_states
                del batch_seg
                del batch_actions
                del batch_rewards
                del batch_action_states
                del batch_next_action_states
                torch.cuda.empty_cache()
                if self.update_type=='soft':
                    for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
   
              # update target network
            if self.update_type=="hard":
                if self.total_updates % self.update_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                    self.target_model.eval()
