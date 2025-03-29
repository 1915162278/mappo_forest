import numpy as np
import math
import geopandas as gpd


# def decode_action(action):  # 解码函数
#     """
#     将整数值解码为具体的动作
#     :param action: 编码后的动作值
#     :return: (tree_type, age_group, cut_rate)
#     """
#     tree_type = action // 110  # 树种类
#     action %= 110
#     age_group = action // 10  # 年龄段
#     cut_rate = action % 10  # 砍伐面积
#     return tree_type, age_group, cut_rate
def biomass_sypress(i):
    # 树种ヒノキ的残材两计算公式)
    biomass = 0.407 * 0.18 * 1.24 * i
    return biomass


def biomass_sedar(i):  # i为森林簿材积
    # 树种スギ的残材量计算公式
    biomass = 0.314 * 0.18 * 1.23 * i
    return biomass


class City:
    def __init__(self, city_name, sum_area):
        self.city_name = city_name
        self.sum_area = sum_area  # 城市面积
        self.forest = []
        self.age_group = 11  # 11个年龄段


    def add_forest(self, ID, species, age, area, volume, group):
        """ 添加林分到城市 """
        forest = Forest(ID, species, age, area, volume, group)
        self.forest.append(forest)

    def get_group_details(self):
        """ 返回该城市所有林分的详细信息 """
        return [forest.get_details() for forest in self.forest]



class Forest:  #
    def __init__(self, ID, species, age, area, volume, group):
        """
        初始化树的属性
        :param ID: 林分id OBJECTID
        :param species: 树种名 (例如 "ヒノキ" 或 "スギ")
        :param age: 树的年龄
        :param area: 森林簿面积
        :param volume: 森林簿材积
        :param group: 所属的group
        """
        self.ID = ID
        self.species = species
        self.age = age
        self.area = area  # 森林簿面积
        self.volume = volume  # 森林簿材积
        self.group = group  # 森林簿材积
        self.is_cut = 0 #判断智能体是否砍伐过这个树

    def get_details(self):
        """ 返回林分的详细信息 """
        return {
            "ID": self.ID,
            "species": self.species,
            "age": self.age,
            "area": self.area,
            "volume": self.volume,
            "group": self.group
        }


class TreeEnv:
    def __init__(self, episode_limit=60, seed=None):
        # 信道数量、设备数量
        self._seed = seed
        self.episode_limit = episode_limit  # 砍伐60年
        self.time_slot = 0 # year

        # 生成智能体
        self.agents = []
        # 市町村和对应的森林面积
        # 市町村和对应的森林面积
        city_data = [
            ("三島市", 999.02),
            ("富士宮市", 10788.84),
            ("富士市", 6042.18),
            ("小山町", 1823.29),
            ("御殿場市", 3375.58),
            ("沼津市", 598.98),
            ("裾野市", 4363.20),
            ("長泉町", 72.46)
        ]
        self.agents = [City(name, area) for name, area in city_data]
        # 读取数据集
        shapefile_path = "./res/fuji250.shp"
        # 数据集列
        # OBJECTID	市町村名		森林簿材積	樹種名_1	                           林齢_1	distance_y	RASTERVALU	geometry
        # 林分       城市名   森林簿面積    森林簿材積    树种名"ヒノキ"用0代表 或 "スギ"用1代表  树的年龄   暂时不用
        self.gdf = gpd.read_file(shapefile_path)
        print(self.gdf.columns)
        # 根据数据集信息，设置智能体属性
        self.n_agents = len(self.agents)
        self.selected_actions = [set() for _ in range(self.n_agents)]
        self.n_actions = len(self.gdf['group'].unique())  # 动作空间维度，0-93

        self.felling_biomass = [0] * self.n_agents #单个智能体当年砍伐的残材量

        self.average_biomass = [1] * self.n_agents #平均残材量
        self.total_biomass = [0] * self.n_agents   #总残材两

        self.group_biomass_agent_sum =0             #多个智能体 总残材量
        self.group_biomass_agent_mean =1            #平均残材量
        # 根据数据集信息，设置智能体属性
        self.initialize_forests()
        # 获取所有智能体的最大林分数量
        self.fixed_forest_count = self.get_max_forest_count()
        self.total_reward=0
        self.sum_felling_biomass=[]

    def initialize_forests(self):
        """ 从数据集中初始化每个城市的group的林分信息 """
        for index, row in self.gdf.iterrows():
            city_name = row['市町村名']
            ID = row['OBJECTID']
            area = row['森林簿面積']
            volume = row['森林簿材積']
            species = row['樹種名_1']
            age = row['林齢_1']
            group = row['group']
            # 查找对应的城市智能体并添加林分
            for agent in self.agents:
                if agent.city_name == city_name:
                    agent.add_forest(ID, species, age, area, volume, group)
                    break

    def get_max_forest_count(self):
        """ 计算所有智能体中的最大林分数量 """
        # 打印所有智能体的林分树
        # for i, agent in enumerate(self.agents):
        #     print(f"智能体 {i + 1} 的林分树数量: {len(agent.forest)}")
        #     # print(f"林分树内容: {agent.forest}")  # 假设 agent.forest 是可打印的
        return max(len(agent.forest) for agent in self.agents)

#     def get_obs_agent(self, agent_id):
#         """ 获取指定智能体的观测空间 """
#         agent = self.agents[agent_id]
#         # 获取该智能体的所有林分信息
#         forest_details = agent.get_group_details()
#         # 提取特征
#
# #####spices1：スギ spices0：ヒノキ的每个年龄段的材积和面积
#         spices1_age_volume = [1]* agent.age_group
#         spices1_age_area = [1]*agent.age_group
#         spices0_age_volume = [1] * agent.age_group
#         spices0_age_area = [1] * agent.age_group
#         for forest in forest_details:
#             if forest['species'] == 'スギ':
#                 if forest['age']<= 10:
#                     spices1_age_volume[0]+=forest['volume']
#                     spices1_age_area[0]+=forest['area']
#                 elif forest['age'] <= 14:
#                     spices1_age_volume[1] += forest['volume']
#                     spices1_age_area[1] += forest['area']
#                 elif forest['age'] <= 19:
#                     spices1_age_volume[2] += forest['volume']
#                     spices1_age_area[2] += forest['area']
#                 elif forest['age'] <= 24:
#                     spices1_age_volume[3] += forest['volume']
#                     spices1_age_area[3] += forest['area']
#                 elif forest['age'] <= 71:
#                     spices1_age_volume[4] += forest['volume']
#                     spices1_age_area[4] += forest['area']
#                 elif forest['age'] <= 90:
#                     spices1_age_volume[5] += forest['volume']
#                     spices1_age_area[5] += forest['area']
#                 elif forest['age'] <= 94:
#                     spices1_age_volume[6] += forest['volume']
#                     spices1_age_area[6] += forest['area']
#                 elif forest['age'] == 95:
#                     spices1_age_volume[7] += forest['volume']
#                     spices1_age_area[7] += forest['area']
#                 elif forest['age'] == 96:
#                     spices1_age_volume[8] += forest['volume']
#                     spices1_age_area[8] += forest['area']
#                 elif forest['age'] == 97:
#                     spices1_age_volume[9] += forest['volume']
#                     spices1_age_area[9] += forest['area']
#                 else :
#                     spices1_age_volume[10] += forest['volume']
#                     spices1_age_area[10] += forest['area']
#
#             else:
#
#                 if forest['age']<= 10:
#                     spices0_age_volume[0]+=forest['volume']
#                     spices0_age_area[0]+=forest['area']
#                 elif forest['age'] <= 14:
#                     spices0_age_volume[1] += forest['volume']
#                     spices0_age_area[1] += forest['area']
#                 elif forest['age'] <= 19:
#                     spices0_age_volume[2] += forest['volume']
#                     spices0_age_area[2] += forest['area']
#                 elif forest['age'] <= 24:
#                     spices0_age_volume[3] += forest['volume']
#                     spices0_age_area[3] += forest['area']
#                 elif forest['age'] <= 71:
#                     spices0_age_volume[4] += forest['volume']
#                     spices0_age_area[4] += forest['area']
#                 elif forest['age'] <= 90:
#                     spices0_age_volume[5] += forest['volume']
#                     spices0_age_area[5] += forest['area']
#                 elif forest['age'] <= 94:
#                     spices0_age_volume[6] += forest['volume']
#                     spices0_age_area[6] += forest['area']
#                 elif forest['age'] == 95:
#                     spices0_age_volume[7] += forest['volume']
#                     spices0_age_area[7] += forest['area']
#                 elif forest['age'] == 96:
#                     spices0_age_volume[8] += forest['volume']
#                     spices0_age_area[8] += forest['area']
#                 elif forest['age'] == 97:
#                     spices0_age_volume[9] += forest['volume']
#                     spices0_age_area[9] += forest['area']
#                 else :
#                     spices0_age_volume[10] += forest['volume']
#                     spices0_age_area[10] += forest['area']
#
#         obs = []
#         obs.append(spices1_age_volume)
#         obs.append(spices1_age_area)
#         obs.append(spices0_age_volume)
#         obs.append(spices0_age_area)
        # for forest in forest_details:
        #
        #     obs.append([
        #         1 if forest['species'] == "ヒノキ" else 0,  # 将树种用0和1表示
        #         forest['age'],
        #         forest['volume']
        #     ])
        # 填充缺少的林分状态
        # while len(obs) < self.fixed_forest_count:
        #     obs.append([0, 0, 0])  # 用0填充

        # 将观测转换为 numpy 数组
        agent_obs = np.array(obs).flatten()  # 所有林分信息展开为一维数组
        # 添加 agent.sum_area 到观测中
        # agent_obs = np.append(agent_obs, agent.sum_area)
        return agent_obs
    def get_obs_agent(self, agent_id):
        """获取指定智能体的观测空间"""
        agent = self.agents[agent_id]
        forest_details = agent.get_group_details()

        # 定义树种和年龄段索引映射
        species_map = {'スギ': [0] * agent.age_group, 'ヒノキ': [0] * agent.age_group}
        age_mapping = [(10, 0), (14, 1), (19, 2), (24, 3), (71, 4), (90, 5), (94, 6), (95, 7), (96, 8), (97, 9),
                       (float('inf'), 10)]

        # 初始化每个树种每个年龄段的材积和面积
        spices_volume = {key: [0] * agent.age_group for key in species_map}
        spices_area = {key: [0] * agent.age_group for key in species_map}

        # 汇总各树种材积和面积
        for forest in forest_details:
            species = forest['species']
            age = forest['age']
            for max_age, idx in age_mapping:
                if age <= max_age:
                    spices_volume[species][idx] += forest['volume']
                    spices_area[species][idx] += forest['area']
                    break

        # 整理为观测数据
        obs = [spices_volume['スギ'], spices_area['スギ'], spices_volume['ヒノキ'], spices_area['ヒノキ']]
        agent_obs = np.array(obs).flatten()  # 展开为一维数组
        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        """ 获取全局状态，即所有智能体观测的合并 """
        return np.concatenate(self.get_obs())

    def reset(self):
        self.time_slot = 0
        self.selected_actions = [set() for _ in range(self.n_agents)]
        self.average_biomass = [1] * self.n_agents
        self.total_biomass = [0] * self.n_agents
        self.felling_biomass = [0] * self.n_agents
        self.group_biomass_agent_sum = 0
        self.group_biomass_agent_mean = 1
        self.sum_felling_biomass=[]
        for agent in self.agents:
            agent.forest = []
        self.initialize_forests()
        return self.get_obs(), self.get_state()

    # 定义未采伐树的材积生长模式（状态转移，智能体没有砍伐时，每年的材积增长按这个来）
    def volume_grow(self, forest):
        if forest.species == 'スギ':
            if forest.age <= 10:
                return forest.volume + 6 * forest.area
            elif forest.age <= 14:
                return forest.volume + 7 * forest.area
            elif forest.age <= 19:
                return forest.volume + 8 * forest.area
            elif forest.age <= 24:
                return forest.volume + 7 * forest.area
            elif forest.age <= 71:
                return forest.volume + 6 * forest.area
            elif forest.age <= 90:
                return forest.volume + 5 * forest.area
            elif forest.age <= 95:
                return forest.volume + 4 * forest.area
            elif forest.age <= 96:
                return forest.volume + 3 * forest.area
            elif forest.age <= 97:
                return forest.volume + 2 * forest.area
            else:
                return forest.volume + 1 * forest.area

        elif forest.species == 'ヒノキ':
            if forest.age <= 24:
                return forest.volume + 4 * forest.area
            elif forest.age <= 94:
                return forest.volume + 5 * forest.area
            elif forest.age <= 95:
                return forest.volume + 4 * forest.area
            elif forest.age <= 96:
                return forest.volume + 3 * forest.area
            elif forest.age <= 97:
                return forest.volume + 2 * forest.area
            else:
                return forest.volume + 1 * forest.area

        return forest.volume

    def volume_grow_cut(self, forest):
        # 计算高度
        height = 32.81 * (1 - np.exp(-0.024 * (forest.age + 1))) ** 0.672

        if forest.species == 'スギ':
            a = 0.054
            b = 1.95
            c = 58.417
            alpha = 28428.78
            beta = 1.082
            delta = 0.000111
            gamma = 2.632
        elif forest.species == 'ヒノキ':
            a = 0.003
            b = 2.948
            c = 39.11
            alpha = 10661.96
            beta = 0.698
            delta = 0.000145
            gamma = 2.55
        else:
            return 0  # 如果物种不在范围内，返回0

        # 计算直径（DBH）
        if height <= 1.2:  # 确保高度合法
            return 0

        DBH = ((1 / (height - 1.2) - 1 / c) ** (-1 / b)) * ((1 / a) ** (1 / b))

        N = alpha * (DBH ** -beta)
        Vs = delta * (DBH ** gamma)

        return Vs * N * forest.area  # 使用森林面积

    def step(self, actions):
        assert len(actions) == len(self.agents)
        # 初始化
        info = {}
        terminated = False
        reward = 0
        self.felling_biomass = [0] * self.n_agents
        self.time_slot += 1
        if self.time_slot == self.episode_limit:
            terminated = True
            info["episode_limit"] = True
        actions_int = [int(a) for a in actions]
        for i, agent in enumerate(self.agents):
            action = int(actions_int[i])  # 智能体选择的group
            self.selected_actions[i].add(action)
            group_forests = [forest for forest in agent.forest if forest.group == action]  # agent选择group
            group_biomass = 0

            ####action选择，计算选择的biomass
            for forest in group_forests:
                forest.is_cut = 1
                # 这边的time——slot不太对，应该是砍伐过后的森林，在林龄生长到20，40的时候砍0.3的材积，并计算biomass。比如说这部分林（v）重新生长到20岁的年份是n，那么第n年的biomass=biomass_seder/sypress（0.3*v） +第n年智能体选择的group的biomass
                # is_special_time = self.time_slot == 20 or self.time_slot == 40 # 间伐年
                # biomass_factor = 0.3 if is_special_time else 1.0
                if forest.species == "ヒノキ":
                    biomass = biomass_sypress(forest.volume)
                else:  # "スギ"
                    biomass = biomass_sedar(forest.volume)
                group_biomass += biomass

                # 更新森林材积
                if self.time_slot < self.episode_limit:  # 未达到最后一步时更新材积
                    # if forest.group == action:  # 砍伐该组的树木
                    forest.age = 0  # 年龄重置为1
                    # new_volume = self.volume_grow_cut(forest)  # 新树苗材积
                    # forest.volume = new_volume
            self.total_biomass[i] += group_biomass  # 每一个agent的total_biomass
            self.felling_biomass[i] = group_biomass
        # self.group_biomass_agent_sum += sum(self.felling_biomass)
        self.group_biomass_agent_mean = sum(self.total_biomass) / self.time_slot
        reward = sum(self.felling_biomass)
        self.sum_felling_biomass.append(reward)
        if self.group_biomass_agent_mean != 0:
            ratio = sum(self.felling_biomass) / self.group_biomass_agent_mean
            if ratio > 1:
                reward *= (1 / ratio) ** 2
            else:
                reward *= ratio ** 2   ####修改

            ######
            # state transition
        for agent in self.agents:
            for forest in agent.forest:
                if forest.is_cut == 0:
                    forest.volume = self.volume_grow(forest)
                else:
                    forest.volume = self.volume_grow_cut(forest)
                forest.age += 1

        return reward, terminated, info


        #     for forest in group_forests:
        #         # 计算生物量
        #         forest.is_cut=1
        #
        #
        #
        #         #这边的time——slot不太对，应该是砍伐过后的森林，在林龄生长到20，40的时候砍0.3的材积，并计算biomass。比如说这部分林（v）重新生长到20岁的年份是n，那么第n年的biomass=biomass_seder/sypress（0.3*v） +第n年智能体选择的group的biomass
        #         # is_special_time = self.time_slot == 20 or self.time_slot == 40 # 间伐年
        #         # biomass_factor = 0.3 if is_special_time else 1.0
        #
        #
        #
        #
        #         #这边计算残材量不需要加成0.3，已经修改
        #         if forest.species == "ヒノキ":
        #             biomass = biomass_sypress(forest.volume )
        #         else:  # "スギ"
        #             biomass = biomass_sedar(forest.volume )
        #         group_biomass += biomass
        #         # 计算奖励
        #         #if self.time_slot != 0:
        #         #     #reward += biomass * (1 - abs(biomass - self.average_bioamss[i]))
        #         #
        #
        #             #奖励更改了，先看看效果如何
        #         #    reward += biomass-abs(biomass-self.average_biomass[i])
        #         # else:
        #         #     reward += biomass
        #
        #
        #
        #         # 更新森林材积
        #         if self.time_slot < self.episode_limit:  # 未达到最后一步时更新材积
        #             # if forest.group == action:  # 砍伐该组的树木
        #                 # 砍伐树木后，更新为新种植的树苗（年龄为1）
        #
        #
        #                 #这里将年龄提前的目的是，volume_grow_cut()这个函数主要和年龄有关，只有年龄变了他才能改变成最初的材积（已修改）
        #                 forest.age = 1  # 年龄重置为1
        #                 new_volume = self.volume_grow_cut(forest)  # 新树苗材积
        #                 forest.volume = new_volume
        #
        #
        #     ######
        #     #state transition
        #     # group=[]
        #     for city in self.agents:
        #         group = []
        #         group=[forest for forest in city.forest if forest.group != action]
        #
        #         for forest in group:
        #             if  forest.is_cut == 0:  # 未砍伐过的树按volume_grow生长
        #
        #                 forest.volume = self.volume_grow(forest)  # 更新未砍伐树木的材积
        #                 forest.age += 1  # 年龄加1
        #             else:
        #                 if (forest.is_cut ==1) & (forest.age!=1):
        #                     forest.volume = self.volume_grow_cut(forest)  # 砍伐过的树按volume_grow生长
        #                     forest.age += 1
        #                 continue
        #
        #     # if self.time_slot != 0:
        #     #     # reward += group_biomass * (1 - abs(biomass - self.average_biomass[i]))
        #     #
        #     #      #奖励更改了，先看看效果如何
        #     #reward += group_biomass - abs(group_biomass - self.average_biomass[i])
        #     # else:
        #     #     reward += group_biomass
        #
        #     # 更新累计总材积
        #
        #     self.total_biomass[i] += group_biomass  #每一个agent的total_biomass
        #     self.felling_biomass[i] = group_biomass
        #     # 计算新的平均材积
        #     #self.average_biomass[i] = self.total_biomass[i] / self.time_slot  #每一个agent的mean biomass
        # if self.time_slot != 0:
        #     self.group_biomass_agent_sum =sum(self.total_biomass)
        #     self.group_biomass_agent_mean = sum(self.total_biomass)/self.time_slot
        #     reward += (sum(self.felling_biomass))*(1-abs(sum(self.felling_biomass)-self.group_biomass_agent_mean)/self.group_biomass_agent_mean)
        #     # reward+= self.group_biomass_agent_mean - (1/60)*(self.group_biomass_agent_sum-self.group_biomass_agent_mean)
        # else:
        #     reward+= self.group_biomass_agent_mean
        # # 状态转移 砍伐过的树木和没砍伐的树木状态更新
        # return reward, terminated, info

    def get_avail_agent_actions(self, agent_id):
        avail_actions = [1 if i not in self.selected_actions[agent_id] else 0 for i in range(self.n_actions)]
        return avail_actions

        # def visualization(self,reward):
        #
        #     print(f'year:{self.time_slot},biomass:{sum(self.felling_biomass)}')
        #     writer.add_scalar('biomass per year',sum(self.felling_biomass),self.time_slot)
        #     # print(f'reward:{reward}')
        #     if self.time_slot>=60:
        #         print (f"total biomass :{self.group_biomass_agent_sum}")
        #         writer.add_scalar('total biomass',self.group_biomass_agent_sum,self.time_slot)
        #         print(f"mean biomass :{self.group_biomass_agent_mean}")
        #         writer.add_scalar('Average biomass', self.group_biomass_agent_mean, self.time_slot)
        #         print(f"total reward :{self.total_reward}")

        # def train_visualization(self):
        #     writer.add_scalar('Average biomass',self.group_biomass_agent_mean,)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def get_env_info(self):
        env_info = {
            "state_shape": len(self.get_state()),
            "obs_shape": len(self.get_obs_agent(0)),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info

    def close(self):
        pass
#
# # 创建环境实例测试
# env = TreeEnv(episode_limit=60)
# # print(env.get_env_info())
# # 重置环境
# observations, state = env.reset()
# # print("Initial Observations:", observations)
# # print("Initial State:", state)
# action = [np.array([4, 53]), np.array([68, 4]), np.array([7, 90]), np.array([82, 82]), np.array([50, 69]),
#           np.array([29, 57]), np.array([57, 61]), np.array([53, 50]), np.array([64, 14]), np.array([31, 88]),
#           np.array([44, 74]), np.array([55, 6]), np.array([74, 84]), np.array([61, 68]), np.array([13, 44]),
#           np.array([14, 55]), np.array([90, 37]), np.array([15, 19]), np.array([72, 11]), np.array([18, 13]),
#           np.array([84, 83]), np.array([6, 40]), np.array([21, 76]), np.array([12, 59]), np.array([69, 56]),
#           np.array([37, 64]), np.array([59, 70]), np.array([5, 21]), np.array([70, 5]), np.array([40, 67]),
#           np.array([80, 72]), np.array([83, 12]), np.array([41, 31]), np.array([91, 91]), np.array([73, 7]),
#           np.array([52, 41]), np.array([10, 58]), np.array([20, 18]), np.array([46, 71]), np.array([36, 46]),
#           np.array([48, 1]), np.array([11, 20]), np.array([2, 15]), np.array([33, 51]), np.array([71, 29]),
#           np.array([42, 73]), np.array([1, 52]), np.array([35, 26]), np.array([23, 28]), np.array([19, 3]),
#           np.array([45, 38]), np.array([76, 36]), np.array([56, 23]), np.array([26, 42]), np.array([25, 65]),
#           np.array([75, 80]), np.array([17, 25]), np.array([58, 78]), np.array([67, 48]), np.array([79, 85])]
#  for step in action:
#     # 随机选择不重复的动作（假设有93个可能的动作，且动作数不能大于总数）
#     actions = step
# # for step in range(60):
# #     # 随机选择不重复的动作（假设有93个可能的动作，且动作数不能大于总数）
# #     actions = np.random.choice(range(env.n_actions), size=env.n_agents, replace=False).tolist()
#     print(actions)
#     # 执行一步
#     reward, terminated, info = env.step(actions)
#
#     # print("Actions:", actions)
#     # print("Reward:", reward)
#     # print("Terminated:", terminated)
#     # print("Info:", info)
#
#     # 获取当前的观测和状态
#     observations = env.get_obs()
#     state = env.get_state()
#     # print("Current Observations:", observations)
#     # print("Current State:", state)
#
#     # 如果到达终止条件，退出循环
#     if terminated:
#         print("Episode ended.")
#         break
