import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# Draw the network topology
def draw_network_topology(network):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(network)  # Positions for all nodes
    nx.draw(network, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black',
            font_weight='bold', edge_color='gray')
    plt.title("Network Topology")
    plt.xlabel(f"Nodes: {network.number_of_nodes()}, Edges: {network.number_of_edges()}")
    plt.show()


class VNF:
    def __init__(self, vnf_id, cpu_req, bw_req, mem_req, max_delay, vi, va, num, y_ug_theta, U, fro_cpu, fro_mem):
        self.vnf_id = vnf_id
        self.cpu_req = cpu_req
        self.bw_req = bw_req
        self.mem_req = mem_req
        self.max_delay = max_delay
        self.vi = vi
        self.va = va
        self.num = num
        self.y_ug_theta = y_ug_theta
        self.U = U
        self.fro_cpu = fro_cpu
        self.fro_mem = fro_mem


def create_network_topology():
    df_nodes = pd.read_csv('./data/20network_topology.csv')
    G = nx.Graph()

    for _, row in df_nodes.drop_duplicates('node_id').iterrows():
        G.add_node(row['node_id'], cpu=row['cpu'], mem=row['mem'])

    for _, row in df_nodes.iterrows():
        G.add_edge(row['node_u'], row['node_w'],
                   link_id=row['link_id'],
                   bw=row['bw'],
                   delay_prop=row['delay_prop'],
                   delay_trans=row['delay_trans'],
                   delay_queue=row['delay_queue'])
    return G


# 动作范围常量
#MAX_NODE_ACTION = 19 # 修改为节点数量
#MAX_LINK_ACTION = 161 # 修改为链路数量
MIN_SLICE_ACTION = 1
MAX_SLICE_ACTION = 31

class VNFEnv:
    def __init__(self, episode_limit=1, seed=None):
        self._seed = seed
        self.episode_limit = episode_limit
        self.time_slot = 0
        self.agents = []
        self.success_vnf = 0
        self.unsuccess_vnf = 0
        df = pd.read_csv('./data/60requests.csv')
        for index, row in df.head(20).iterrows():  # 读取5个智能体
            vnf = VNF(
                vnf_id=row['vnf_id'],
                cpu_req=row['cpu_req'],
                bw_req=row['BW_req'],
                mem_req=row['mem_req'],
                max_delay=row['max_delay'],
                vi=row['VI'],
                va=row['VA'],
                num=row['Num'],
                y_ug_theta=row['y_ug_theta'],
                U=row['U'],
                fro_cpu=row['fro_cpu'],
                fro_mem=row['fro_mem']
            )
            # Print VNF and all attributes
            print(f"VNF ID: {vnf.vnf_id}, CPU: {vnf.cpu_req}, BW: {vnf.bw_req}, "
                  f"Memory: {vnf.mem_req}, Max Delay: {vnf.max_delay}, "
                  f"VI: {vnf.vi}, VA: {vnf.va}, Num: {vnf.num}, "
                  f"y_ug_theta: {vnf.y_ug_theta}, U: {vnf.U}, "
                  f"fro_cpu: {vnf.fro_cpu}, fro_mem: {vnf.fro_mem}")
            self.agents.append(vnf)

        self.n_agents = len(self.agents)
        self.network = create_network_topology()
        # 画出一个网络拓扑图表示这个network
        # draw_network_topology(self.network)
        num_nodes = self.network.number_of_nodes()
        num_edges = self.network.number_of_edges()
        self.max_node_action = self.network.number_of_nodes()-1
        self.max_link_action = self.network.number_of_edges()-1
        self.network_state = np.zeros(num_nodes * 3 + num_edges * 3)

        self.validate_network()
        self.n_actions = self.network.number_of_nodes() * self.network.number_of_edges() * 31

    def encode_action(self,node_action, link_action, slice_action):
        # 确保输入在正确的范围内
        if not (0 <= node_action <= self.max_node_action):
            raise ValueError(f"节点动作必须在范围 [0, {self.max_node_action}] 内")
        if not (0 <= link_action <= self.max_link_action):
            raise ValueError(f"链路动作必须在范围 [0, {self.max_link_action}] 内")
        if not (MIN_SLICE_ACTION <= slice_action <= MAX_SLICE_ACTION):
            raise ValueError(f"切片动作必须在范围 [{MIN_SLICE_ACTION}, {MAX_SLICE_ACTION}] 内")

        # 编码动作
        return (node_action * (self.max_link_action + 1) * (MAX_SLICE_ACTION)) + \
               (link_action * (MAX_SLICE_ACTION)) + \
               (slice_action - MIN_SLICE_ACTION)

    def decode_action(self,encoded_action):
        # 解码动作
        slice_action = (encoded_action % (MAX_SLICE_ACTION)) + MIN_SLICE_ACTION
        link_action = (encoded_action // (MAX_SLICE_ACTION)) % (self.max_link_action + 1)
        node_action = encoded_action // ((self.max_link_action + 1) * (MAX_SLICE_ACTION))

        return node_action, link_action, slice_action

    def reset(self):
        self.time_slot = 0
        self.success_vnf = 0
        self.unsuccess_vnf = 0
        # 重新创建网络拓扑，以获取初始状态
        self.network = create_network_topology()
        num_nodes = self.network.number_of_nodes()
        num_edges = self.network.number_of_edges()

        # 初始化 network_state 为正确的大小，并填充初始状态
        self.network_state = np.zeros(num_nodes * 3 + num_edges * 3)
        for i, node in enumerate(self.network.nodes(data=True)):
            scale_factor = np.random.uniform(0.9, 1.1)  # 生成随机缩放因子
            self.network_state[i * 3] = node[0]  # node ID
            self.network_state[i * 3 + 1] = node[1]['cpu'] * scale_factor  # CPU, 缩放
            self.network_state[i * 3 + 2] = node[1]['mem'] * scale_factor  # Memory, 缩放

        edge_start = 3 * num_nodes
        for i, edge in enumerate(self.network.edges(data=True)):
            scale_factor_bw = np.random.uniform(0.9, 1.1)  # 为带宽生成随机缩放因子
            self.network_state[edge_start + i * 3] = edge[2]['link_id']  # link ID
            self.network_state[edge_start + i * 3 + 1] = edge[2]['bw'] * scale_factor_bw  # Bandwidth, 缩放
            # 延迟通常不参与缩放，因为它是固定的物理属性
            self.network_state[edge_start + i * 3 + 2] = edge[2]['delay_prop'] + edge[2]['delay_trans'] + edge[2]['delay_queue']  # Total delay

        return self.get_obs(), self.get_state()

    def update_network(self, agent, node, link_id):
        # 找到选中节点在节点列表中的索引
        node_index = list(self.network.nodes()).index(node)

        # 更新节点信息
        self.network_state[node_index * 3] = node
        self.network_state[node_index * 3 + 1] -= (agent.cpu_req + agent.fro_cpu)  # 减少CPU资源
        self.network_state[node_index * 3 + 2] -= (agent.mem_req + agent.fro_mem)  # 减少内存资源# 检查资源是否满足
        if (self.network_state[node_index * 3 + 1] >= 0) and (self.network_state[node_index * 3 + 2] >= 0):
            # 如果 CPU 和内存资源都满足
            self.success_vnf += 1
        else:
            # 如果存在一种资源不满足
            self.unsuccess_vnf += 1
        # 查找对应link_id的边
        edge = None
        for u, v in self.network.edges():
            if self.network.edges[u, v]['link_id'] == link_id:
                edge = (u, v)
                break
        # print("edge", edge)  # 调试信息
        if edge is None:
            raise ValueError(f"Link ID {link_id} not found in the network.")

        # 使用新的find_edge_index方法找到边的索引
        edge_index = self.find_edge_index(*edge)  # 修改：使用新方法
        edge_start = 3 * self.network.number_of_nodes()

        # 更新边信息
        self.network_state[edge_start + edge_index * 3] = self.network.edges[edge]['link_id']
        self.network_state[edge_start + edge_index * 3 + 1] -= agent.bw_req  # 减少带宽资源

        # 更新相邻节点和边的信息
        for neighbor in self.network.neighbors(node):
            neighbor_index = list(self.network.nodes()).index(neighbor)

            # 如果相邻节点未初始化，则初始化它
            if self.network_state[neighbor_index * 3] == 0:
                self.network_state[neighbor_index * 3:neighbor_index * 3 + 3] = [
                    neighbor,
                    self.network.nodes[neighbor]['cpu'],
                    self.network.nodes[neighbor]['mem']
                ]

            # 获取连接到相邻节点的边
            edge = (node, neighbor) if (node, neighbor) in self.network.edges() else (neighbor, node)
            edge_index = self.find_edge_index(*edge)  # 修改：使用新方法

            # 如果边未初始化，则初始化它
            if self.network_state[edge_start + edge_index * 3] == 0:
                edge_data = self.network.edges[edge]
                self.network_state[edge_start + edge_index * 3:edge_start + edge_index * 3 + 3] = [
                    edge_data['link_id'],
                    edge_data['bw'],
                    edge_data['delay_prop'] + edge_data['delay_trans'] + edge_data['delay_queue']
                ]

    # 新添加的方法
    def find_edge_index(self, u, v):
        """
        找到给定边(u, v)在边列表中的索引。

        参数:
        u, v: int，边的两个端点

        返回:
        int: 边在边列表中的索引

        如果边不存在，抛出ValueError。
        """
        edges = list(self.network.edges())
        if (u, v) in edges:
            return edges.index((u, v))
        elif (v, u) in edges:
            return edges.index((v, u))
        else:
            raise ValueError(f"在网络中未找到边 ({u}, {v})")

    def get_agent_inf(self, agent_id):
        vnf = self.agents[agent_id]
        return np.array([
            vnf.cpu_req,
            vnf.bw_req,
            vnf.mem_req,
            vnf.max_delay,
            vnf.vi,
            vnf.va,
            vnf.num,
            vnf.y_ug_theta,
            vnf.U,
            vnf.fro_cpu,
            vnf.fro_mem
        ])

    def get_obs_agent(self, agent_id):  # 观测空间，智能体属性+拓扑属性
        obs_inf = self.get_agent_inf(agent_id)
        agent_obs = np.concatenate((self.network_state, obs_inf))
        return agent_obs

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):  # 状态空间，所有智能体属性+拓扑属性
        agents_info = np.concatenate([self.get_agent_inf(i) for i in range(self.n_agents)])
        state = np.concatenate((self.network_state, agents_info))
        return state

    def get_avail_agent_actions(self, agent_id):
        # 设定所有动作都是可用的
        avail_actions = [1] * self.n_actions
        return avail_actions

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def step(self, actions):
        # print(f"Actions: {actions}, Length: {len(actions)}")
        # print(f"Number of agents: {len(self.agents)}")
        assert len(actions) == len(self.agents)
        info = {}  # 测评的其它指标放这里
        terminated = False
        reward = 0
        delay = 0
        self.time_slot += 1
        if self.time_slot == self.episode_limit:
            terminated = True
            info["episode_limit"] = True

        for i, agent in enumerate(self.agents):
            action = int(actions[i])
            d_actions = self.decode_action(action)  # 多维离散动作解码
            # print("d_actions", d_actions)
            node = int(d_actions[0])
            link_id = int(d_actions[1])
            slice_type = int(d_actions[2])
            # 找到对应的边
            edge_tuple = None
            for u, v in self.network.edges():
                if self.network.edges[u, v]['link_id'] == link_id:
                    edge_tuple = (u, v)
                    break

            if edge_tuple is None:
                raise ValueError(f"Link ID {link_id} not found in the network.")  # 链路不存在

            self.update_network(agent, node, link_id)

            # 使用 edge_tuple 计算延迟
            edge_delay = self.network.edges[edge_tuple]['delay_prop'] + \
                         self.network.edges[edge_tuple]['delay_trans'] + \
                         self.network.edges[edge_tuple]['delay_queue']
            delay += edge_delay
            # 计算隔离度奖励
            iso_v = agent.vi + agent.U * (agent.va - agent.vi) / (slice_type + 1)  # 加1避免除以0
            isolation_reward = iso_v * agent.num  # 乘以num，因为num代表VNF的数量
            # 计算切片类型收益
            slice_type_result = np.bitwise_and(int(agent.y_ug_theta), int(slice_type))
            slice_type_reward = bin(slice_type_result).count('1')
            # 合并所有奖励
            agent_reward = agent.max_delay - edge_delay + isolation_reward + slice_type_reward
            reward += agent_reward
            # 如果内存、时延、CPU不满足匹配，或者选到了没激活的网络或者链路给予惩罚
        info["success_vnf"] = self.success_vnf
        info["unsuccess_vnf"] = self.unsuccess_vnf
        return reward, terminated, info

    def validate_network(self):
        # 打印基本信息
        print(f"Number of nodes: {self.network.number_of_nodes()}")
        print(f"Number of edges: {self.network.number_of_edges()}")

        # 打印节点信息
        print("Nodes and their attributes:")
        for node, data in self.network.nodes(data=True):
            print(f"Node ID: {node}, Attributes: {data}")

        # 打印边信息
        print("Edges and their attributes:")
        for u, v, data in self.network.edges(data=True):
            print(f"Edge from {u} to {v}, Attributes: {data}")

    def get_env_info(self):
        env_info = {
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs_agent(0)),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# # test
# env = VNFEnv()
# print(env.get_env_info())
# obs, state = env.reset()
# print(f"Initial observation shape: {np.array(obs).shape}")
# print(f"Initial state shape: {state.shape}")
# print(f"Number of actions: {env.n_actions}")
# # 示例：执行随机动作
# specified_actions = [
#     (0, 1, 2),  # Action for agent 1
#     (1, 39, 3),  # Action for agent 2
#     (2, 27, 1),  # Action for agent 3
#     (0, 0, 1),  # Action for agent 4
#     (1, 39, 2),  # Action for agent 5
# ]
#
# # Encode each agent's action once
# encoded_actions = [env.encode_action(*actions) for actions in specified_actions]
# print("encoded_actions", encoded_actions)
# for _ in range(3):
#     # Step the environment with the encoded actions
#     reward, terminated, info = env.step(encoded_actions)
#     print(f"Reward: {reward}, Terminated: {terminated}")
#     if terminated:
#         break
