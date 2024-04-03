from DSSE import DroneSwarmSearch
from DSSE import Actions
from DSSE.tests.drone_policy import policy

def init_drone_swarm_search(grid_size=20, render_mode="human", render_grid=True, render_gradient=True,
                            vector=(3.5, -0.5), disperse_constant=5, 
                            timestep_limit=300, person_amount=1, person_initial_position=None,
                            drone_amount=1, drone_speed=10,
                            drone_probability_of_detection=0.9):

    if person_initial_position is None:
        person_initial_position = (grid_size - 1, grid_size - 1)
        
    return DroneSwarmSearch(
            grid_size=grid_size,
            render_mode=render_mode,
            render_grid=render_grid,
            render_gradient=render_gradient,
            vector=vector,
            disperse_constant=disperse_constant,
            timestep_limit=timestep_limit,
            person_amount=person_amount,
            person_initial_position=person_initial_position,
            drone_amount=drone_amount,
            drone_speed=drone_speed,
            drone_probability_of_detection=drone_probability_of_detection,
        )

# env = DroneSwarmSearch(
#     grid_size=40,
#     render_mode="human",
#     render_grid=True,
#     render_gradient=True,
#     vector=(3.2, 3.1),
#     disperse_constant=5,
#     timestep_limit=200,
#     person_amount=5,
#     person_initial_position=(10, 10),
#     drone_amount=2,
#     drone_speed=10,
#     drone_probability_of_detection=0.9,
#     pre_render_time = 0,
# )

env = init_drone_swarm_search(person_amount=5, person_initial_position=(10, 10), drone_amount=1)

print(env.get_agents())

# from DSSE import Actions

# def calculate_action(drone_position, target_position):
#     drone_x, drone_y = drone_position
#     target_x, target_y = target_position
    
#     # Verificar se o drone e a pessoa estão na mesma posição
#     if drone_x == target_x and drone_y == target_y:
#         return Actions.SEARCH.value  # Realizar a ação de busca
    
#     # Calcular a diferença nas coordenadas x e y
#     diff_x = target_x - drone_x
#     diff_y = target_y - drone_y
    
#     # Decidir a ação com base na diferença nas coordenadas
#     if diff_x < 0 and diff_y < 0:
#         return Actions.UP_LEFT.value
#     elif diff_x > 0 and diff_y < 0:
#         return Actions.UP_RIGHT.value
#     elif diff_x < 0 and diff_y > 0:
#         return Actions.DOWN_LEFT.value
#     elif diff_x > 0 and diff_y > 0:
#         return Actions.DOWN_RIGHT.value
#     elif diff_x < 0:
#         return Actions.LEFT.value
#     elif diff_x > 0:
#         return Actions.RIGHT.value
#     elif diff_y < 0:
#         return Actions.UP.value
#     else:
#         return Actions.DOWN.value



# print(env.agents_positions)

# import math

# def distance(point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2

#     # Calculando a diferença nas coordenadas x e y
#     diff_x = x2 - x1
#     diff_y = y2 - y1

#     # Calculando a distância euclidiana
#     distance = math.sqrt(diff_x**2 + diff_y**2)
#     return distance

# def policy(obs, agents):
#     actions = {}
#     for agent in agents:  # Considerando que você tenha um único drone
#         # Obter as posições atuais de todos os náufragos
#         castaways_positions = [person.get_position() for person in env.get_persons()]
        
#         # Supondo que drone_position obtenha a posição atual do drone
#         drone_position = env.agents_positions['drone0']
        
#         # Encontrar o náufrago mais próximo (isso é apenas um esqueleto, precisa de uma função para calcular)
#         nearest_castaway = min(castaways_positions, key=lambda pos: distance(drone_position, pos))
#         print(drone_position, nearest_castaway)
        
#         # Decidir a ação com base na posição do náufrago mais próximo (isso precisa ser implementado)
#         action = calculate_action(drone_position, nearest_castaway)
        
#         actions[agent] = action
    
#     return actions

opt = {
    "drones_positions": [(0, 0), (0, 5), (0, 10), (0, 15)],
}
observations, info = env.reset()

# observations, info = env.reset(options={"drones_positions": [(8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12),
#                 (10, 8), (10, 9), (10, 11), (10, 12), (10, 10),
#                 (11, 8), (11, 9), (11, 10), (11, 11), (11, 12),
#                 (12, 8), (12, 9), (12, 10), (12, 11), (12, 12)]})

rewards = 0
done = False
while not done:
    actions = policy(observations, env.get_agents(), env=env)
    observations, reward, _, done, info = env.step(actions)
    rewards += sum(reward.values())
    done = any(done.values())

_ = env.reset()
print(rewards)
