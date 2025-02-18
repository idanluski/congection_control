import math
import random
import networkx as nx
import matplotlib.pyplot as plt

k=0.0001
ALPHA=500
iteration=100
MODE = "DUAL"



lirning_rate_dict = { 1: 0.0001,2:0.001,500:1}
factor = {1:1,2:2,20:2000000}
k = lirning_rate_dict[ALPHA]

# Define nodes and edges
users = [f'a{i}' for i in range(6)] #0,1,2,3,4,5,
edges = [(f'a{i}', f'a{i+1}') for i in range(5)]  # Linear connections


# Define capacities (all are 1)
capacities = {edge: 1 for edge in edges}

# Define paths
paths = {
    ('a0', 'a5'): [('a0','a1'),('a1','a2'),('a2','a3'),('a3','a4'),('a4','a5')],
    **{(f'a{i}', f'a{i+1}'): [(f'a{i}', f'a{i+1}')] for i in range(5)}
}



# # Define edges (links) and capacities
# edges = [('S1', 'A1'), ('A1', 'A2'), ('A2', 'D1'), 
#          ('S2', 'B1'), ('B1', 'B2'), ('B2', 'D2'),
#          ('A1', 'B1'),  # Shared link between A1 and B1
#          ('B2', 'A2'),  # Another shared link
#          ('D1', 'D2')]  # Direct link between D1 and D2

# capacities = {('S1', 'A1'): 100, ('A1', 'A2'): 80, ('A2', 'D1'): 90,
#               ('S2', 'B1'): 120, ('B1', 'B2'): 70, ('B2', 'D2'): 100,
#               ('A1', 'B1'): 50, ('B2', 'A2'): 60, ('D1', 'D2'): 110}

# # Define users (nodes)
# users = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',]

# # Define paths between source-destination pairs
# paths = {
#     ('S1', 'D1'): [('S1', 'A1'),('A1', 'A2'),('A2', 'D1')],  # Main path
#     ('S2', 'D2'): ['S2', 'B1', 'B2', 'D2'],  # Another path
#     ('S1', 'D2'): ['S1', 'A1', 'B1', 'B2', 'D2'],  # Shared path (A1 -> B1 -> B2)
#     ('S2', 'D1'): ['S2', 'B1', 'A1', 'A2', 'D1'],  # Another shared path
# }


def utility(x, a):
    """
    Computes the expression: (x^(1-a)) / (1-a), with a special case for a=1 returning log(x).

    Args:
        x (float): The base value (must be positive for log).
        a (float): The exponent modifier.

    Returns:
        float: Computed value of the expression.
    """
    if x <= 0:
        raise ValueError("x must be positive for valid exponentiation and logarithm.")

    if a == 1:
        return math.log(x)  # Special case when a=1
    
    return (x ** (1 - a)) / (1 - a)

def f_func(rate,c):
    """
    Computes the expression: 1 / (c - x).

    Args:
        x (float): The base value (must be different from c).
        c (float): The constant value.

    Returns:
        float: Computed value of the expression.
    """
    
    return  factor[ALPHA]*(rate / c) **2
    # return  ALPHA*(rate / c) ** 2
 
        
       
       
       

def x_r_tag(G, x_r):
    """
    Computes the derivative of the utility function with respect to x_r.
    
    """
    u_tag = min(G.nodes[x_r]["rate"] ** -ALPHA, 100000000000) 
    path_xr = G.nodes[x_r]["path"]
    sum = 0
    for edge in path_xr:
        sum += G.edges[edge]['cost']

    return u_tag - sum


def update_link_cost(G,xr):
    """
    Updates the cost of the link that x_r is using.
    """
    path_xr = G.nodes[xr]["path"]

    for edge in path_xr:
        sum_of_rates = G.edges[edge]['sum_of_rate']
        capacity = G.edges[edge]['capacity']
        f = f_func(sum_of_rates, capacity)
        G.edges[edge]['cost'] = f

def  next_x_r(G, xr, learning_rate):
    """
    Computes the next value of x_r using the gradient descent method.
    """
    xr_t = x_r_tag(G, xr)
    previus = G.nodes[xr]['rate']
    next =  max(previus + learning_rate * xr_t ,0.1)
    path_of_xr = G.nodes[xr]["path"]
    for p in path_of_xr:
        sum_of_rates = G.edges[p]['sum_of_rate'] - previus + next
        G.edges[p]['sum_of_rate'] = sum_of_rates
    G.nodes[xr]["rate"] = next
    update_link_cost(G, xr)



def plus_function(G,edge):
    lamda = G.edges[edge]['lamda']
    y = G.edges[edge]['sum_of_rate']
    c = G.edges[edge]['capacity']
    if lamda > 0:
        return y - c
    return max(y-c, 0)




def update_lamde(G, edge):
    lamda_old = G.edges[edge]['lamda']
    lamda_d = k * plus_function(G, edge)  # k is your dual step size
    
    # NOTE: Add lamda_d instead of subtracting it
    lamda_new = lamda_old*0.9 + 0.1*lamda_d
    
    # Update Q and x for each flow using this edge
    for x_r in G.edges[edge]['x_r']:
        # Remove old contribution, add new
        G.nodes[x_r]['Q'] += lamda_new - lamda_old
        
        # Then update the rate from Q (for α=1, x = 1/Q)
        # Just as you had before:
        next_x = (G.nodes[x_r]['Q'] + 1e-6)**(-1/ALPHA)
        
        # Update sum_of_rate on the path
        old_rate = G.nodes[x_r]['rate']
        for p in G.nodes[x_r]["path"]:
            G.edges[p]['sum_of_rate'] = G.edges[p]['sum_of_rate'] - old_rate + next_x
        
        # Finally set the new rate
        G.nodes[x_r]['rate'] = next_x
    
    # Update lambda on this edge
    G.edges[edge]['lamda'] = lamda_new
      




def primal_distribute(edges, capacities, users, paths):
    
    """
    Plots a network graph with edges, nodes, and paths.

    Args:
        edges (list of tuples): List of edges (links) in the form [(u, v), (v, w), ...].
        capacities (dict): Dictionary with edge capacities {(u, v): capacity, ...}.
        users (list): List of user nodes.
        paths (dict): Dictionary with source-destination pairs as keys and paths as values.
    """
    G = nx.Graph()

    
    # Add nodes
    for user in users:
        G.add_node(user)
        G.nodes[user]['rate'] = 0.0001
        G.nodes[user]['Q'] = 0
        G.nodes[user]['path'] = set()
        for key in paths.keys():
            if key[0] == user:
                G.nodes[user]['path'].update(paths[key])
        
    G.nodes[users[-1]]['rate'] = 0
    # Add edges with capacities
    for edge in edges:
        G.add_edge(edge[0], edge[1])
        G.edges[edge]['capacity'] = capacities.get(edge, 0)
        G.edges[edge]['x_r'] = set()
        G.edges[edge]['lamda'] = 0.1
        G.edges[edge]['cost'] = 0
        G.edges[edge]['sum_of_rate'] = 0

    for user in users:
        path = G.nodes[user]['path']
        for edge in path:
            G.nodes[user]['Q'] += G.edges[edge]['lamda']


    
    # Update edges with x_r values
    for key, path in paths.items():
        for p in path:
            edge = p
            if 'x_r' not in G.edges[edge]:  # Ensure 'x_r' exists
                G.edges[edge]['x_r'] = set()
            G.edges[edge]['x_r'].add(key[0])  # Add the first element of key
            G.edges[edge]['sum_of_rate'] += G.nodes[key[0]]['rate']
            print(G.edges[edge]['x_r'])
    # Define node positions
    pos = nx.spring_layout(G, seed=42)  # Layout for visualization

    # Draw nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10)

    # # Draw edge labels (capacities)
    # edge_labels = {(u, v): f"{capacities.get((u, v), 0)}" for u, v in edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Highlight paths in different colors
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    path_labels = {}
    
    # for i, (pair, path) in enumerate(paths.items()):
    #     edges_in_path = [path[j] for j in range(len(path))]
    #     nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color=colors[i % len(colors)], width=2.5)

        # # Add text annotation for paths
        # mid_edge = edges_in_path[len(edges_in_path) // 2]
        # mid_pos = [(pos[mid_edge[0]][0] + pos[mid_edge[1]][0]) / 2, 
        #            (pos[mid_edge[0]][1] + pos[mid_edge[1]][1]) / 2]
        
        #path_text = " → ".join(path)
        #path_labels[mid_edge] = path_text

    rates_statistic = {user: [] for user in users}
    users = users[:-1]
    for i in range(iteration):
        if MODE == "PRIMAL":
            random_node = random.choice(users)
            next_x_r(G, random_node, k)
            
            # Compute the rates for each user
            for user in users:
                rates_statistic[user].append(G.nodes[user]['rate'])
        else:
            random_edge = random.choice(edges)
            update_lamde(G, random_edge)
            # Compute the rates for each user
            for user in users:
                rates_statistic[user].append(G.nodes[user]['rate'])

        
    # Compute the final rates
        # print("Rates:", rates_statistic)



    # Plot the convergence of rates over time


    iteration_range = range(iteration)
    rate_convergence = [[G.nodes[user]['rate'] for user in users] for _ in iteration_range]

    plt.figure()
    for i, user in enumerate(users):
        plt.plot(iteration_range, rates_statistic[user], label=user)
    plt.xlabel("Iteration")
    plt.ylabel("Rate")
    plt.title("Rate Convergence")
    plt.legend()
    plt.show()

primal_distribute(edges, capacities, users, paths)