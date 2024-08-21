import random
from re import X
import gym
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import collections

from flow.benchmarks.figureeight9 import flow_params
from flow.benchmarks.figureeight_all import flow_params as flow_params_fine_tuning
from flow.utils.registry import make_create_env
import torch.optim as optim

from tensorboardX import SummaryWriter
import gc


buffer_size = 100000
collision = 0
totaldone = 0
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done, action_log): 
        self.buffer.append((state, action, reward, next_state, done, action_log)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, action_log = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, action_log

    def size(self): 
        return len(self.buffer)


def clip_grad_norm(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


def get_distance_matrix(infos):
    xy_list = np.array(list(infos.values())) 
    distances_matrix = np.sqrt(np.sum((xy_list[:, np.newaxis, :] - xy_list[np.newaxis, :, :]) ** 2, axis=-1))
    return distances_matrix


def get_communi_vehicle(vehicle_id, distance_matrix, agents_num, radius):   
    vehicle_distances = distance_matrix[vehicle_id]
    nearby_vehicles = [(i, vehicle_distances[i]) for i in range(agents_num) if vehicle_distances[i] <= radius and i != vehicle_id]   
    nearby_vehicles_sorted = sorted(nearby_vehicles, key=lambda x: x[1])
    return nearby_vehicles_sorted


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        
    def mu(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        return mu
   
    def std(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        std = F.softplus(self.fc_std(x)) + 1e-5
        return std

    def dist(self, state):
        mu = self.mu(state)
        std = self.std(state)
        return Normal(mu, std)

    def forward(self, x, deterministic=False): 
        mu = self.mu(x)
        std = self.std(x)
        dist = Normal(mu, std)
        
        if deterministic:
            normal_sample = mu
        else:
            normal_sample = dist.rsample()  
        
        log_prob = dist.log_prob(normal_sample)
        action= torch.tanh(normal_sample) 
        log_prob = log_prob - (2. * (math.log(2.) - normal_sample - F.softplus(-2. * normal_sample)))
        action = action * self.action_bound
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
    

class SACContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, segments, replica):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  
        self.replay_buffer = ReplayBuffer(buffer_size)
            
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.segments = segments
        self.replica = replica
        
        self.data = [] 
        self.actor_param = []
        self.critic_1_param = []
        self.critic_2_param = []
        self.beta = torch.tensor(0, dtype=torch.float).to(self.device)
        self.policy_advantage = torch.tensor(0, dtype=torch.float).to(self.device)


    def get_param(self):
        self.actor_param = [x.data for x in self.actor.parameters()]


    def take_action(self, state, deterministic):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, log_prob = self.actor(state, deterministic)
        return action.item(), log_prob.item() 

    
    def calc_target(self, rewards, next_states, dones):  
        with torch.no_grad():
            next_actions, log_prob = self.actor(next_states) 
            td_target = rewards + self.gamma * (torch.min(
            self.target_critic_1(next_states, next_actions),
            self.target_critic_2(next_states, next_actions)
        ) + self.log_alpha.exp() * (-log_prob)) * (1 - dones)
        return td_target
    

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)


    def get_policy_adv(self, model, batch_size, gamma):
        b_s, b_a, b_r, b_ns, b_d, b_a_log  = self.replay_buffer.sample(batch_size)
        transition_dict = {'states': b_s, 'actions': b_a, 'actions_log':b_a_log}
        
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        actions_log = torch.tensor(transition_dict['actions_log'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        
        with torch.no_grad():
            dist_tilde = model.actor.dist(states) 
            log_prob_tilde = dist_tilde.log_prob(actions)
            log_prob_tilde = torch.clamp(log_prob_tilde, -20, 0.0)
            pi_tilde_prob = torch.exp(log_prob_tilde)
            tilde_entropy = dist_tilde.entropy()

            dist_self = self.actor.dist(states) 
            log_prob_self = dist_self.log_prob(actions) 
            log_prob_self = torch.clamp(log_prob_self, -20.0, 0.0)
            pi_self_prob = torch.exp(log_prob_self)
            self_entropy = dist_self.entropy()
            
            q1_value = self.critic_1(states, actions)
            q2_value = self.critic_2(states, actions)
            Q_value = torch.min(q1_value, q2_value)

            off_actions_prob = torch.exp(actions_log)

            entropy_minus = tilde_entropy - self_entropy

            policy_adv = ((pi_tilde_prob - pi_self_prob) / (off_actions_prob + 1e-5)) * Q_value + self.log_alpha.exp() * entropy_minus

        self.policy_advantage = torch.mean(policy_adv)
        Coff = (2 * torch.max(policy_adv) * gamma) / ((1 - gamma) ** 2)
        self.beta = self.get_metric(model, states, actions, Coff)


    def get_metric(self, model, states, actions, Coff):
        appr = True 
        self.get_param()
        model.get_param()
        param_div = [x.data - k.data for k, x in zip(self.actor_param, model.actor_param)]
        param_div_vector = torch.cat([x.view(-1) for x in param_div]).unsqueeze(-1)
        dist_self = self.actor.dist(states) 
        log_prob_self = dist_self.log_prob(actions)
        log_prob_self = torch.clamp(log_prob_self, -20.0, 0.0)
        prob = torch.exp(log_prob_self)
        if appr:
            actor_loss = -log_prob_self.mean()         
            saved_grads = [param.grad.clone() if param.grad is not None else None for param in self.actor.parameters()]
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad = [param.grad.data.clone() for param in self.actor.parameters()]
            grad_vector = torch.cat([grad.view(-1) for grad in actor_grad]).unsqueeze(-1)
            FIM = grad_vector @ grad_vector.t()
            beta = torch.sqrt((2 * (self.policy_advantage) / (Coff * (param_div_vector.t() @ FIM @ param_div_vector)))).squeeze()

            for param, saved_grad in zip(self.actor.parameters(), saved_grads):
                if saved_grad is not None:
                    param.grad = saved_grad
        else:
            FIM = None
            batch = states.size(0)
            for i in range(batch):
                self.actor_optimizer.zero_grad()
                actor_loss = -log_prob_self[i]
                actor_loss.backward(retain_graph=True)

                fp16_params = []
                for param in self.actor.parameters():
                    if param.grad is not None:
                        grad_float16 = param.grad.view(-1).detach().to(dtype=torch.float16)
                        param.grad = None 
                        fp16_params.append(grad_float16)
                sample_grad = torch.cat(fp16_params).unsqueeze(-1)
                torch.cuda.empty_cache()
                outer_product = prob[i].to(dtype=torch.float16) * sample_grad @ sample_grad.t()
                if FIM is None:
                    FIM = outer_product
                else:
                    FIM.add_(outer_product)
                del outer_product
                gc.collect()
                torch.cuda.empty_cache()
            del actor_loss
            FIM /= batch
            beta = torch.sqrt((2 * self.policy_advantage.to(dtype=torch.float16) / (Coff.to(dtype=torch.float16) * (param_div_vector.t() @ FIM @ param_div_vector)))).squeeze()
            beta = beta.clone().detach()
            beta = beta.to(dtype=torch.float32)

        return beta
              

    def mix_policy(self, actor_param):
        for k, x in zip(actor_param, self.actor.parameters()):
            x.data.copy_((1 - self.beta) * x.data + self.beta * k.data)


    def get_reshaped_param(self):
        actor_param = [np.array(x.data.cpu()) for x in self.actor.parameters()] 
        return np.array(actor_param, dtype=object)


    def get_segments(self, target_model_weights, p, segments):
        flat_m = []
        shape_list = []
        for x in target_model_weights:
            shape_list.append(x.shape) 
            flat_m.extend(list(x.flatten()))
        seg_length = len(flat_m) // segments + 1 
        return flat_m[p*seg_length:(p+1)*seg_length], shape_list 


    def reconstruct(self,flat_m,shape_list):
        result = []
        current_pos = 0
        for shape in shape_list:
            total_number = 1
            for i in shape:
                total_number *= i 
            result.append(np.array(flat_m[current_pos:current_pos+total_number]).reshape(shape))
            current_pos += total_number
        return np.array(result, dtype=object)
    

    def cache_param(self, actor_param):
        for k, x in zip(actor_param, self.actor.parameters()):
            x.data.copy_(k.data)


    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)


        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        clip_grad_norm(self.critic_1.parameters(), 0.5)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        clip_grad_norm(self.critic_2.parameters(), 0.5)
        self.critic_2_optimizer.step()


        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()


        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def modify(obs, agents_num):
    modified_obs = []
    if agents_num ==14:
        for i in range(agents_num):
            rl_loc = obs[i + 14]
            rl_speed = obs[i]

            if i != 0:
                follower_loc = obs[i - 1 + 14]
                follower_speed = obs[i - 1]
            else:
                follower_loc = obs[13 + 14]
                follower_speed = obs[13]

            leader_loc = obs[(i + 1) % 14 + 14]
            leader_speed = obs[(i + 1) % 14]
            modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

    elif agents_num == 9:
        rl_loc = obs[15] 
        rl_speed = obs[1]

        follower_loc = obs[14]
        follower_speed = obs[0]

        leader_loc = obs[16] 
        leader_speed = obs[2]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

        rl_loc = obs[17]
        rl_speed = obs[3]

        follower_loc = obs[16]
        follower_speed = obs[2]

        leader_loc = obs[18] 
        leader_speed = obs[4]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

        rl_loc = obs[18]
        rl_speed = obs[4]

        follower_loc = obs[17] 
        follower_speed = obs[3]

        leader_loc = obs[19] 
        leader_speed = obs[5]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

        rl_loc = obs[19] 
        rl_speed = obs[5]

        follower_loc = obs[18]
        follower_speed = obs[4]

        leader_loc = obs[20] 
        leader_speed = obs[6]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

        rl_loc = obs[20] 
        rl_speed = obs[6]

        follower_loc = obs[19] 
        follower_speed = obs[5]

        leader_loc = obs[21] 
        leader_speed = obs[7]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

        rl_loc = obs[22] 
        rl_speed = obs[8]

        follower_loc = obs[21] 
        follower_speed = obs[7]

        leader_loc = obs[23] 
        leader_speed = obs[9]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

        rl_loc = obs[23] 
        rl_speed = obs[9]

        follower_loc = obs[22] 
        follower_speed = obs[8]

        leader_loc = obs[24] 
        leader_speed = obs[10]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

        rl_loc = obs[24] 
        rl_speed = obs[10]

        follower_loc = obs[23] 
        follower_speed = obs[9]

        leader_loc = obs[25] 
        leader_speed = obs[11]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))


        rl_loc = obs[26] 
        rl_speed = obs[12]

        follower_loc = obs[25] 
        follower_speed = obs[11]

        leader_loc = obs[27] 
        leader_speed = obs[13]
        modified_obs.append(np.array([rl_loc, rl_speed, follower_loc, follower_speed, leader_loc, leader_speed]))

    return np.array(modified_obs)


def test_agent(env, agents, repeat, horizon, agents_num):
    global totaldone
    global collision
    score_stochastic = 0
    for _ in range(repeat):
        global_state = env.reset()
        obs = modify(global_state, agents_num)
        done = False
        ho = 0
        
        while not done:
            actions = []
            actions = [agents[i].take_action(obs[i], deterministic=False)[0] for i in range(agents_num)]
            next_global_state, reward, done, _ = env.step(actions)
            next_obs = modify(next_global_state, agents_num)
            score_stochastic += reward/horizon
            obs = next_obs
            ho += 1
            if done:
                collision += (ho != horizon)
                totaldone += 1
                break
    test_score = score_stochastic/repeat
    score_stochastic = 0.0 
    return test_score


def save_model(policy_net, critic_net_1, critica_net_2, filepath):
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'critic_net_1_state_dict': critic_net_1.state_dict(),
        'critic_net_2_state_dict': critica_net_2.state_dict()
    }, filepath)


def load_model(policy_net, critic_net_1, critica_net_2, filepath):
    checkpoint = torch.load(filepath)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    critic_net_1.load_state_dict(checkpoint['critic_net_1_state_dict'])
    critica_net_2.load_state_dict(checkpoint['critic_net_2_state_dict'])


def simulate_packet_loss(segment, loss_rate):
    for i in range(len(segment)):
        if random.random() < loss_rate:
            segment[i] = 0
    return segment


def train(actor_lr, critic_lr, num_episodes, tau, batch_size, comm_interval, test_interval, repeat, cuda, segments, replica, max_step, radius, head, fine_tuning, checkpoint, loadfile, loss_rate, testOnly):
    if fine_tuning is True:
        create_env, env_name = make_create_env(flow_params_fine_tuning, version=0)
        agents_num = 14
    else: 
        create_env, env_name = make_create_env(flow_params, version=0)
        agents_num = 9
    
    env = create_env()
    state_dim = 6
    action_dim = 1
    action_bound = env.action_space.high[0]  

    alpha_lr = 3e-4  
    horizon = 1500 
    hidden_dim = 256
    gamma = 0.99

    return_list = []
    test_score = []
    print_interval = []
    iterations = 0
    eps = np.finfo(np.float32).eps
    num_sample = 50 

    policy_adv_judge = 0.0 
    test_score_stochastic =[]

    success_mix = 0
    total_mix = 0

    communication_matrix = np.zeros((agents_num, agents_num)) 
    MixSuccess_matrix = np.zeros((agents_num, agents_num))
    distance_matrix = np.zeros((agents_num, agents_num))

    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda:{}".format(cuda)) if torch.cuda.is_available() else torch.device(
        "cpu")

    agents = [SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                        actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                        gamma, device, segments, replica) for _ in range(2*agents_num)]

    writer = SummaryWriter(head + f"rewards")
    

    if fine_tuning is True:
        load_model(agents[-1].actor, agents[-1].critic_1, agents[-1].critic_2, loadfile)

    for i in range(agents_num):
        agents[i].actor.load_state_dict(agents[-1].actor.state_dict())
        agents[i].critic_1.load_state_dict(agents[-1].critic_1.state_dict())
        agents[i].critic_2.load_state_dict(agents[-1].critic_2.state_dict())
        agents[i].target_critic_1.load_state_dict(agents[i].critic_1.state_dict())
        agents[i].target_critic_2.load_state_dict(agents[i].critic_2.state_dict())
    

    # begin simulation
    for m in range(10):
        with tqdm(total=int(num_episodes/10),desc='Iteration %d' % m) as pbar:
            for i_episode in range(int(num_episodes/10)): 
                episode_return = 0
                global_state = env.reset()
                obs = modify(global_state, agents_num)
                done = False

                if testOnly is False:     
                    while not done:
                        for step in range(max_step):
                            actions = []
                            actions_log = []
                            
                            for i in range(agents_num):
                                state = obs[i]
                                action, action_log = agents[i].take_action(state, deterministic=False)
                                actions.append(action)
                                actions_log.append(action_log)
                            next_global_state, reward, done, infos = env.step(actions)
                            
                            episode_return += reward / horizon

                            next_obs = modify(next_global_state, agents_num)

                            for i in range(agents_num):
                                agents[i].replay_buffer.add(obs[i], actions[i], reward, next_obs[i], done, actions_log[i])
                            obs = next_obs
                            if done:
                                break

                        if agents[0].replay_buffer.size() > 2 * batch_size : 
                            
                            for i in range(agents_num):
                                b_s, b_a, b_r, b_ns, b_d, _= agents[i].replay_buffer.sample(batch_size)
                                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                                agents[i].update(transition_dict)
                            

                            if iterations % comm_interval == 0:
                                distance_matrix = get_distance_matrix(infos)
                                
                                for i in range(agents_num):                                 
                                    comm_range_with_distances = get_communi_vehicle(i, distance_matrix, agents_num, radius)                            
                                    segments_ = min(agents[i].segments, len(comm_range_with_distances) )
                                    replica_ = min(agents[i].replica, len(comm_range_with_distances) )

                                    for re in range(replica_):
                                        actor_reconstruct_list = []
                                        target_list = []
                                        random.shuffle(comm_range_with_distances)  
                                        target_agents = np.array([t[0] for t in comm_range_with_distances]) 
                                        distances = np.array([t[1] for t in comm_range_with_distances]) 
                                                                        
                                        for p in range(segments_):
                                            target_agent = target_agents[p] 
                                            target_list.append(target_agent)                                    
                                            target_actor_weights = agents[target_agent].get_reshaped_param()
                                            seg, shape_record = agents[i].get_segments(target_actor_weights, p, segments_)
                                            
                                            if loss_rate != 0 :
                                                seg = simulate_packet_loss(seg, loss_rate) 

                                            actor_reconstruct_list.extend(seg) 
            
                                        avg_actor_sum = np.array(agents[i].reconstruct(actor_reconstruct_list, shape_record))

                                        for ii in range(len(avg_actor_sum)):
                                            len_in = len(avg_actor_sum[ii])
                                            for j in range(len_in):
                                                avg_actor_sum[ii][j] = avg_actor_sum[ii][j].tolist()
                                            avg_actor_sum[ii] = torch.from_numpy(avg_actor_sum[ii]).float().to(device)
                                        
                                        cache_num = i + agents_num
                                        agents[cache_num].cache_param(avg_actor_sum) 
                                        agents[i].get_policy_adv(agents[cache_num], num_sample, gamma) 
                                        total_mix += 1
                                        policy_adv_judge = agents[i].policy_advantage.cpu().detach().numpy().item()                          
                                        agents[i].beta = torch.clamp(agents[i].beta , 0.0, 0.8)
                                        
                                        for each_tar in target_list:
                                            communication_matrix[i][each_tar] += 1

                                        if policy_adv_judge > 0.0: 
                                            agents[i].mix_policy(avg_actor_sum) 
                                            for each_tar in target_list:
                                                MixSuccess_matrix[i][each_tar] += 1                                           
                                            success_mix += 1
                        
                            iterations += 1

                    writer.add_scalar("reward_training", episode_return, (num_episodes/10 * m + i_episode+1))   
                    return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * m + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

                if (num_episodes/10 * m + i_episode+1) % test_interval == 0  :
                    test_score = test_agent(env, agents, repeat, horizon, agents_num)
                    test_score_stochastic.append(test_score)
                    print_interval.append(num_episodes/10 * m + i_episode+1)
                    writer.add_scalar("reward_test", test_score, (num_episodes/10 * m + i_episode+1))  

                
                if (num_episodes/10 * m + i_episode+1) % 100 == 0 :
                    if checkpoint is True:
                        checkpoint_filepath = head + f"checkpoint_step{(num_episodes/10 * m + i_episode+1)}.pth"
                        save_model(agents[0].actor, agents[0].critic_1, agents[0].critic_2, checkpoint_filepath)
                    
    writer.close()


    np.fill_diagonal(communication_matrix, 0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im1 = axes[0].imshow(communication_matrix, cmap='viridis', interpolation='nearest')
    axes[0].set_title('Communication HeatMap')
    axes[0].set_xlabel('Agent i')
    axes[0].set_ylabel('Agent j')
    axes[0].set_xticks(np.arange(agents_num))
    axes[0].set_yticks(np.arange(agents_num))
    fig.colorbar(im1, ax=axes[0], label='Communication Times')

  
    np.fill_diagonal(MixSuccess_matrix, 0)
    im2 = axes[1].imshow(MixSuccess_matrix, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Successful Mixture HeatMap')
    axes[1].set_xlabel('Agent i')
    axes[1].set_ylabel('Agent j')
    axes[1].set_xticks(np.arange(agents_num))
    axes[1].set_yticks(np.arange(agents_num))
    fig.colorbar(im2, ax=axes[1], label='Effective Mixture Times')

    plt.savefig(head + "Heatmap.png")


if __name__ == "__main__":
    import argparse
    from configparser import ConfigParser

    cfg = ConfigParser()
    cfg.read("config.ini")

    actor_lr = float(cfg["train_parameters"]["actor_lr"])
    critic_lr = float(cfg["train_parameters"]["critic_lr"])
    num_episodes = float(cfg["train_parameters"]["num_episodes"])


    # get local iteration inforamtion
    parser = argparse.ArgumentParser()

    parser.add_argument("-tau", type=float, required=False, default=0.001, help="soft update rate")
    parser.add_argument("-batch_size", type=int, required=False, default=256, help="batch size number")
    parser.add_argument("-comm_interval", type=int, required=False, default=8, help="communication interval")
    parser.add_argument("-test_interval", type=int, required=False, default=10, help="test interval")
    parser.add_argument("-repeat", type=int, required=False, default=1, help="test numbers")
    parser.add_argument("-cuda", type=int, required=False, default=0, help="gpu idx")
    parser.add_argument("-seg", type=int, required=False, default=4, help="segment number")
    parser.add_argument("-re", type=int, required=False, default=3, help="replica number")
    parser.add_argument("-maxstep", type=int, required=False, default=10, help="maxstep")
    parser.add_argument("-radius", type=int, required=False, default=90, help="communication range")
    parser.add_argument("-output", type=str, required=False, default = None, help="filepath of outputs")
    parser.add_argument("-fine_tuning", type=bool, required=False, default = False, help="whether load the converged model to fine tuning in new environments")
    parser.add_argument("-checkpoint", type=bool, required=False, default = False, help="whether save the final model")
    parser.add_argument("-loadfile", type=str, required=False, default = None, help="load converged model file")
    parser.add_argument("-loss_rate", type=float, required=False, default = 0, help="Communication loss rate.")
    parser.add_argument("-testOnly", type=bool, required=False, default = False, help="whether only test, not train the policy.")
   
    args = parser.parse_args()

    tau = args.tau
    batch_size = args.batch_size
    comm_interval = args.comm_interval
    test_interval = args.test_interval
    repeat = args.repeat
    cuda = args.cuda
    segments = args.seg
    replica = args.re
    max_step = args.maxstep
    radius = args.radius
    head = args.output
    fine_tuning = args.fine_tuning
    checkpoint = args.checkpoint
    loadfile = args.loadfile
    testOnly = args.testOnly
    loss_rate = args.loss_rate


    train(actor_lr, 
                    critic_lr, 
                    num_episodes, 
                    tau, 
                    batch_size, 
                    comm_interval, 
                    test_interval, 
                    repeat, 
                    cuda, 
                    segments, 
                    replica, 
                    max_step, 
                    radius, 
                    head, 
                    fine_tuning, 
                    checkpoint,
                    loadfile,
                    loss_rate,
                    testOnly)