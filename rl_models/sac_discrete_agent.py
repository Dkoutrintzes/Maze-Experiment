import torch
import numpy as np
from rl_models.networks_discrete import update_params, Actor, Critic, ReplayBuffer
import torch.nn.functional as F
from collections import deque
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class DiscreteSACAgent:
    def __init__(self,args = None, config=None, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, buffer_max_size=1000000, tau=0.005,
                 update_interval=1, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2,
                 chkpt_dir=None,load_file=None, target_entropy_ratio=0.4,participant_name=None,ID='First'):
        self.args = args
        self.ID = ID
        
        if config is not None:
            self.config = config
            self.mode = config['Experiment']['mode']
            self.args.actor_lr = config['SAC']['alpha']
            self.args.critic_lr = config['SAC']['beta']
            self.args.hidden_size = [config['SAC']['layer1_size'],config['SAC']['layer2_size']]
            self.args.hidden_sizes = [config['SAC']['layer1_size'],config['SAC']['layer2_size']]
            self.args.tau = config['SAC']['tau']
            self.args.gamma = config['SAC']['gamma']
            self.args.batch_size = config['SAC']['batch_size']
            self.args.buffer_size = args.buffer_size
            if self.ID == 'First':
                self.freeze_agent = self.config['SAC']['freeze_agent']
            elif self.ID == 'Second':
                self.freeze_agent = self.config['SAC']['freeze_second_agent']
            
            self.chkpt_dir = config['SAC']['chkpt']
            if self.ID == 'First':
                self.load_file = os.path.join(config['SAC']['load_file'],self.args.agent_type)
            elif self.ID == 'Second':
                self.load_file = os.path.join(config['SAC']['load_second_file'], self.args.agent_type)

            if self.args.ppr and self.ID == 'Expert':
                self.load_file = self.args.expert_policy
                self.freeze_agent = True

        else:
            self.args.actor_lr = alpha
            self.args.critic_lr = beta
            self.args.hidden_size = [layer1_size,layer2_size]
            self.args.hidden_sizes = [layer1_size,layer2_size]
            self.args.tau = tau
            self.args.gamma = gamma
            self.args.batch_size = batch_size
            self.load_file = load_file
            self.chkpt_dir = chkpt_dir

        
        self.update_interval = update_interval
        self.buffer_max_size = buffer_max_size
        self.env = env
        self.p_name = participant_name

        self.step = 0
        self.soft_update_every = 1

        self.args.state_shape = input_dims[0] 
        self.args.action_shape  = args.num_actions
        print("action_shape",self.args.action_shape)
        print("state_shape",self.args.state_shape)
        
        self.model_name = config['SAC']['model_name']

        # Saving arrays
        self.alpha_history = []
        
        self.q1_history = []
        self.q2_history = []

        self.q1_loss_history = []
        self.q2_loss_history = []

        self.policy_history = []
        self.policy_loss_history = []

        self.entropy_history = []
        self.entropy_loss_history = []

        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.next_states_history = []
        self.dones_history = []
        self.transition_infos = []

        self.actor = Actor(self.args.state_shape, self.args.action_shape, self.args.hidden_sizes,name=self.model_name, chkpt_dir=self.chkpt_dir,load_file = self.load_file).to(device)
        self.critic = Critic(self.args.state_shape, self.args.action_shape, self.args.hidden_sizes,name=self.model_name, chkpt_dir=self.chkpt_dir,load_file = self.load_file).to(device)
        self.target_critic = Critic(self.args.state_shape, self.args.action_shape, self.args.hidden_sizes,name=self.model_name, chkpt_dir=self.chkpt_dir,load_file = self.load_file).to(
            device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, eps=1e-8)
        self.critic_q1_optim = torch.optim.Adam(self.critic.qnet1.parameters(), lr=self.args.critic_lr, eps=1e-8)
        self.critic_q2_optim = torch.optim.Adam(self.critic.qnet2.parameters(), lr=self.args.critic_lr, eps=1e-8)

        if self.args.auto_alpha:
            self.target_entropy =  0.50 * np.log(np.prod(self.args.action_shape))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.alpha_lr, eps=1e-8)
        else:
            self.log_alpha = torch.tensor(self.args.alpha, requires_grad=True, device=device)
            self.alpha = self.args.alpha

        
        self.memory = ReplayBuffer(args)

        self.average_reward = []
        self.average_q = []
        self.average_next_q = []
        self.average_target_q = []
        self.average_alpha = []
        self.average_z = []

        self.action1_prob = []
        self.action2_prob = []
        self.action3_prob = []

        self.average_entropy = []
        self.idx = 0
        self.step = 0

    def update_params(self, optim, loss):
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()


    def learn(self,block_nb):
 
        states, actions, rewards, states_, dones,transition_info = self.memory.sample(block_nb,self.args.batch_size)
        self.average_reward.append(np.mean(rewards))
        #print(actions)
        states = torch.from_numpy(states).float().to(device)
        states_ = torch.from_numpy(states_).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        batch_transitions = states, actions, rewards, states_, dones

        weights = 1.  # default
        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch_transitions, weights)
        policy_loss, entropies, q1, q2, action_probs = self.calc_policy_loss(batch_transitions, weights)

        self.average_entropy.append(entropies.mean().item())


        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        
        self.update_params(self.critic_q1_optim, q1_loss)
        self.update_params(self.critic_q2_optim, q2_loss)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        self.update_params(self.actor_optim, policy_loss)


        if self.args.auto_alpha:
            entropy_loss = self.calc_entropy_loss(entropies, weights)

            self.alpha_optim.zero_grad()
            entropy_loss.backward(retain_graph=True)
            self.alpha_optim.step()
            self.entropy_loss_history.append(entropy_loss.item())
            self.alpha = self.log_alpha.exp()



        self.alpha_history.append(self.log_alpha.exp().item())

        self.q1_loss_history.append(q1_loss.item())
        self.q2_loss_history.append(q2_loss.item())

        self.policy_loss_history.append(policy_loss.item())

        self.entropy_history.append(entropies.mean().item())

        # if cycle_i % 25 == 0:
        #     print("Policy loss: ", policy_loss)
        #     print("Q1 loss: ", q1_loss)
        #     print("Q2 loss: ", q2_loss)
        #     print("Alpha: ", self.alpha)
        self.step += 1
        if self.step % self.args.update_target_every == 0:
            self.soft_update_target()
            
        return  policy_loss.item(), q1_loss.item(), q2_loss.item(), self.log_alpha.exp().item()

    def supervised_learn(self,block_nb):
        states, actions, rewards, states_, dones,transition_info = self.memory.sample(block_nb,self.args.batch_size)
        action_tensor = []
        for action in actions:
            temp = [0,0,0]
            temp[int(action)] = 1
            action_tensor.append(temp)

        actions = action_tensor

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)

        states = torch.from_numpy(states).float().to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]


        self.actor_optim.zero_grad()
        action_probs = self.actor(states)
        #print(action_probs)
        #print(action_probs)
        #print(actions)
        loss = F.cross_entropy(action_probs, actions.squeeze(1))
        loss.backward()
        self.actor_optim.step()
        


        #print("Supervised learning loss: ", loss.item())
        return loss.item()


    def add_point(self):
        self.alpha_hisotry.append(0)
        self.alpha_hisotry.append(1)

    def get_alpha_history(self):
        return self.alpha_history

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.args.tau * param + (1 - self.args.tau) * target_param)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions)  # select the Q corresponding to chosen A
        curr_q2 = curr_q2.gather(1, actions)
        self.average_q.append(min(curr_q1.mean().item(),curr_q2.mean().item()))
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            action_probs = self.actor(next_states)
            z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
            log_action_probs = torch.log(action_probs + z)

            next_q1, next_q2 = self.target_critic(next_states)
            self.average_next_q.append(min(next_q1.mean().item(),next_q2.mean().item()))
            self.average_z.append(z.mean().item())
            self.average_alpha.append(self.log_alpha.exp().item())

            for a in range(3):
                temp = action_probs[:,a].unsqueeze(1)
                if a == 0:
                    self.action1_prob.append(temp.mean().item())
                elif a == 1:
                    self.action2_prob.append(temp.mean().item())
                elif a == 2:
                    self.action3_prob.append(temp.mean().item())
            # next_q = (action_probs * (
            #     torch.min(next_q1, next_q2) - self.alpha * log_action_probs
            # )).mean(dim=1).view(self.memory_batch_size, 1) # E = probs T . values

            next_q = action_probs * (torch.min(next_q1, next_q2) - self.alpha * log_action_probs)
            next_q = next_q.sum(dim=1)
            #print((1 - dones).mean())
            target_q = rewards + (1 - dones) * self.args.gamma * (next_q)
            return target_q.unsqueeze(1)

 
    def calc_critic_loss(self, batch, weights):
        target_q = self.calc_target_q(*batch)
        self.average_target_q.append(target_q.mean().item())
        # TD errors for updating priority weights
        # errors = torch.abs(curr_q1.detach() - target_q)
        errors = None
        mean_q1, mean_q2 = None, None

        # We log means of Q to monitor training.
        # mean_q1 = curr_q1.detach().mean().item()
        # mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        # q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        # q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        action_probs = self.actor(states)
        z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
        log_action_probs = torch.log(action_probs + z)

        # with torch.no_grad():
        # Q for every actions to calculate expectations of Q.
        # q1, q2 = self.critic(states)
        # q = torch.min(q1, q2)
        with torch.no_grad():
            q1, q2 = self.critic(states)

        # minq = torch.min(q1, q2)
        # inside_term = alpha * log_action_probs - minq
        # policy_loss = (action_probs * inside_term).mean()

        # Expectations of entropies.
        entropies = - torch.sum(action_probs * log_action_probs, dim=1)
        
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()  # avg over Batch

        return policy_loss, entropies, q1, q2, action_probs

    def calc_entropy_loss2(self, pi_s, log_pi_s):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.

        inside_term = - self.alpha * (log_pi_s + self.target_entropy).detach()
        entropy_loss = (pi_s * inside_term).mean()
        return entropy_loss

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach()
            * weights)
        return entropy_loss
    
    def can_learn(self,block_nb):
        if self.args.dqfd:
            splits = [1,0.8,0.6,0.4,0.2,0.1,0.05,0,0,0,0]
            blk_req = int(self.args.batch_size*(1- splits[block_nb]))
            if blk_req < self.memory.get_size():
                return True
            else:
                return False
            
        else:
            #print(self.memory.get_size())
            #print(self.args.buffer_size)
            if self.args.batch_size < self.memory.get_size():
                return True
            else:
                return False

    def save_models(self,block_number):
        if self.chkpt_dir is not None:
            print('.... saving models ....')
            self.actor.save_checkpoint(block_number)
            self.critic.save_checkpoint(block_number)
            #self.target_critic.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        #self.target_critic.load_checkpoint()
        self.soft_update_target()
        print('.... models loaded ....')
    
    def collect_data(self):
        return_data = {}

        # Collect data for plotting.
        return_data['q1'] = self.q1_history
        return_data['q2'] = self.q2_history

        return_data['policy'] = self.policy_history
        return_data['policy_loss'] = self.policy_loss_history

        return_data['q1_loss'] = self.q1_loss_history
        return_data['q2_loss'] = self.q2_loss_history
        
        return_data['entropy_loss'] = self.entropy_loss_history
        return_data['entropies'] = self.entropy_history
        return_data['alpha'] = self.alpha_history

        return_data['states'] = self.states_history
        return_data['actions'] = self.actions_history
        return_data['rewards'] = self.rewards_history
        return_data['next_states'] = self.next_states_history
        return_data['dones'] = self.dones_history



        # reseting arrays
        self.alpha_history = []

        self.q1_history = []
        self.q2_history = []

        self.q1_loss_history = []
        self.q2_loss_history = []

        self.policy_history = []
        self.policy_loss_history = []

        self.entropy_history = []
        self.entropy_loss_history = []

        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.next_states_history = []
        self.dones_history = []

        return return_data
    
    def return_settings(self):
        actor_lr = self.args.actor_lr
        critic_lr = self.args.critic_lr
        hidden_size = self.args.hidden_size
        tau = self.args.tau
        gamma = self.args.gamma
        batch_size = self.args.batch_size

        if self.args.auto_alpha:
            target_entropy = self.target_entropy
            log_alpha = self.log_alpha.item()
            alpha_lr = self.args.alpha_lr
        else:
            target_entropy = None
            log_alpha = self.log_alpha.item()
            alpha_lr = None
        
        return self.ID,actor_lr,critic_lr,alpha_lr,hidden_size,tau,gamma,batch_size,target_entropy,log_alpha,self.freeze_agent
    
    def flatten_curbs(self,data):
        temp = []
        window = deque(maxlen=20)
        for i in range(len(data)):
            window.append(data[i])
            if len(window) == 20:
                temp.append(np.mean(window))
        return temp
    def save_test_stats(self):
        if not os.path.exists('test_stats'):
            os.makedirs('test_stats')
            
        self.average_reward = self.flatten_curbs(self.average_reward)
        self.average_q = self.flatten_curbs(self.average_q)
        self.average_next_q = self.flatten_curbs(self.average_next_q)
        self.average_target_q = self.flatten_curbs(self.average_target_q)
        self.average_alpha = self.flatten_curbs(self.average_alpha)
        self.average_z = self.flatten_curbs(self.average_z)
        self.action1_prob = self.flatten_curbs(self.action1_prob)
        self.action2_prob = self.flatten_curbs(self.action2_prob)
        self.action3_prob = self.flatten_curbs(self.action3_prob)
        self.average_entropy = self.flatten_curbs(self.average_entropy)
         
        save_csv('test_stats/'+'average_reward.csv',self.average_reward)
        save_csv('test_stats/'+'average_q.csv',self.average_q)
        save_csv('test_stats/'+'average_next_q.csv',self.average_next_q)
        save_csv('test_stats/'+'average_target_q.csv',self.average_target_q)
        save_csv('test_stats/'+'average_alpha.csv',self.average_alpha)
        save_csv('test_stats/'+'average_z.csv',self.average_z)
        save_csv('test_stats/'+'action1_prob.csv',self.action1_prob)
        save_csv('test_stats/'+'action2_prob.csv',self.action2_prob)
        save_csv('test_stats/'+'action3_prob.csv',self.action3_prob)
        save_csv('test_stats/'+'entropy.csv',self.average_entropy)


        self.average_reward = []
        self.average_q = []
        self.average_next_q = []
        self.average_target_q = []
        self.average_alpha = []
        self.average_z = []
        self.action1_prob = []
        self.action2_prob = []
        self.action3_prob = []
        self.average_entropy = []


import csv
def save_csv(path,data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow([row])