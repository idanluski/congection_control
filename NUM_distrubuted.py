import math
import random
import networkx as nx
import matplotlib.pyplot as plt


# Confirm the file was saved
import os
#os.path.abspath("/Users/iluski/Desktop/project_network_security/graph_generate/graph.png")
filename = "/Users/ohav1/OneDrive/Desktop/Betihut/graph_generate/graph.png"

k = 0.0001

ALPHA = 20
iteration = 200000
MODE = "PRIMAL"
#MODE = "DUAL"
ROUTE = "D" #OR B/D
DYNAMIC = True


learning_rate_dict_P = {1: 0.001, 2: 0.0001, 20: 0.000000002}
learning_rate_dict = {1: 0.0001, 2: 0.001,  float('inf'): 0.001} #dual
factor = {1:1,2:2,20:2000000}
if MODE == "PRIMAL":
    k = learning_rate_dict_P[ALPHA]
else:
    k = learning_rate_dict[ALPHA]

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

def find_shortest_path_dijkstra(G, source, target, weight_key='cost'):
    return nx.shortest_path(G, source=source, target=target, weight=lambda u, v, d: d.get(weight_key, 1), method="dijkstra")


def find_shortest_path_bellman_ford(G, source, target, weight_key='cost'):

    return nx.shortest_path(G, source=source, target=target, weight=lambda u, v, d: d.get(weight_key, 1), method='bellman-ford')


def dymenic_path(G, xr, method, count):
    dest = G.nodes[xr]['dest']
    if ROUTE == "D":
        path = find_shortest_path_dijkstra(G, xr, dest, weight_key=method)
    else:
        path = find_shortest_path_bellman_ford(G, xr, dest, weight_key=method)
    path_edges = list(zip(path[:-1], path[1:]))

    prev_path = G.nodes[xr]['path']
    prev_rate = G.nodes[xr]['rate']
    if (path_edges != list(prev_path)) and (count < 10):
        print(f"node {xr} changed path from {list(prev_path)} to {path_edges}")
        count = count + 1

    for edge in prev_path:
        G.edges[edge]['sum_of_rate'] -= prev_rate
        G.edges[edge]['x_r'].discard(xr)

    G.nodes[xr]['path'] = path_edges

    for edge in path_edges:
        G.edges[edge]['x_r'].add(xr)
    if MODE == "DUAL":
        min_rate = float('inf')
        G.nodes[xr]['Q'] = sum(G.edges[edge]['lamda'] for edge in G.nodes[xr]['path'])
        Q = G.nodes[xr]['Q']
        # Compute rate based on α
        if ALPHA == 1:  # Proportional fairness
            next_x = 1 / max(Q, 1e-6)

        elif ALPHA == float('inf'):  # Max-min fairness
            next_x = 1 / max(Q, 1e-6)
            min_rate = min(min_rate, next_x)  # Track min rate in the network

        else:  # General α-fairness
            next_x = Q ** (-1 / ALPHA)
        G.nodes[xr]['rate'] = next_x
        for edge in path_edges:
            G.edges[edge]['sum_of_rate'] += G.nodes[xr]['rate']
    return count



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
    if ALPHA >2:
        u_tag = min(G.nodes[x_r]["rate"] ** -ALPHA, 100000000)
    else:
        u_tag = min(G.nodes[x_r]["rate"] ** -ALPHA, 1000)
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
        if DYNAMIC:
            sum_of_rates = G.edges[p]['sum_of_rate'] + next
        else:
            sum_of_rates = G.edges[p]['sum_of_rate'] - previus + next
        G.edges[p]['sum_of_rate'] = sum_of_rates
    G.nodes[xr]["rate"] = next
    update_link_cost(G, xr)


def plus_function(G, edge):
    y = G.edges[edge]['sum_of_rate']
    c = G.edges[edge]['capacity']
    return max(y - c, 0)  # Ensure non-negative value


def update_lamde(G, edge):
    """
    Updates the lambda values (prices) in the dual algorithm for different α values.
    """
    lamda_old = G.edges[edge]['lamda']
    lamda_d = k * plus_function(G, edge)  # k is your dual step size

    # Update lambda value
    lamda_new = max(lamda_old + lamda_d, 0)  # Ensure non-negative lambda

    # Step 1: Compute new rates based on alpha fairness
    min_rate = float('inf')  # Track the minimum rate for α=∞ fairness
    updated_rates = {}  # Store updated rates before applying
    if not DYNAMIC: #same path only one edge changes
        for x_r in G.edges[edge]['x_r']:
            # Remove old contribution, add new lambda update
            G.nodes[x_r]['Q'] += lamda_new - lamda_old
            Q = G.nodes[x_r]['Q']

            # Compute rate based on α
            if ALPHA == 1:  # Proportional fairness
                next_x = 1 / max(Q, 1e-6)

            elif ALPHA == float('inf'):  # Max-min fairness
                next_x = 1 / max(Q, 1e-6)
                min_rate = min(min_rate, next_x)  # Track min rate in the network

            else:  # General α-fairness
                next_x = Q ** (-1 / ALPHA)

            updated_rates[x_r] = next_x  # Store new rate

        # Step 2: If α=∞, force all flows on this bottleneck to the same min rate
        if ALPHA == float('inf'):
            for x_r in updated_rates:
                updated_rates[x_r] = min_rate  # Enforce equal rates on bottlenecked users

        # Step 3: Apply the new rates and update sum_of_rate on links
        for x_r, new_rate in updated_rates.items():
            old_rate = G.nodes[x_r]['rate']
            for p in G.nodes[x_r]["path"]:
                G.edges[p]['sum_of_rate'] = G.edges[p]['sum_of_rate'] - old_rate + new_rate
            G.nodes[x_r]['rate'] = new_rate  # Apply final rate

        # Set the new lambda value for the edge
    G.edges[edge]['lamda'] = lamda_new


def main(edges, capacities, users, paths):
    
    """
    Plots a network graph with edges, nodes, and paths.

    Args:
        edges (list of tuples): List of edges (links) in the form [(u, v), (v, w), ...].
        capacities (dict): Dictionary with edge capacities {(u, v): capacity, ...}.
        users (list): List of user nodes.
        paths (dict): Dictionary with source-destination pairs as keys and paths as values.
    """
    G = nx.Graph()
    count = 0
    
    # Add nodes
    for user in users:
        G.add_node(user)
        G.nodes[user]['rate'] = 0.0001
        G.nodes[user]['Q'] = 0
        G.nodes[user]['path'] = set()
        for key in paths.keys():
            if key[0] == user:
                G.nodes[user]['path'].update(paths[key])

    user_set = set(users)
    user_send = set()
    for p in paths.keys():
        user_send.add(p[0])
    user_not_send = user_set - user_send  
    for user in user_not_send:  
        G.nodes[user]['rate'] = 0 

    # Add edges with capacities
    for edge in edges:
        G.add_edge(edge[0], edge[1])
        G.edges[edge]['capacity'] = capacities.get(edge, 0)
        G.edges[edge]['x_r'] = set()
        G.edges[edge]['lamda'] = 0.1
        G.edges[edge]['cost'] = 0
        G.edges[edge]['sum_of_rate'] = 0

    for user in user_send:
        path = G.nodes[user]['path']
        for edge in path:
            G.nodes[user]['Q'] += G.edges[edge]['lamda']

    for xr in user_send:
        G.nodes[xr]['dest'] = random.choice([user for user in users if user != xr])


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

    rates_statistic = {user: [] for user in user_send}
    users = list(user_send)
    for i in range(iteration):
        if MODE == "PRIMAL":
            random_node = random.choice(users)
            if DYNAMIC:
                count = dymenic_path(G, random_node, 'cost', count)
            next_x_r(G, random_node, k)
            
            # Compute the rates for each user
            for user in users:
                rates_statistic[user].append(G.nodes[user]['rate'])
        else:
            random_edge = random.choice(edges)
            update_lamde(G, random_edge)
            if DYNAMIC:
                for xr in users:
                    count = dymenic_path(G, xr, 'lamda', count)


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
    plt.title(f"Rate Convergence: mode {MODE}learning rate {k} ALPHA {ALPHA}")
    plt.legend()
    plt.show()





def generate_random_graph(num_users=6, num_edges=8, seed=None):
    """
    Generates a connected random graph and extracts its components.
    
    Args:
        num_users (int): Number of users (nodes).
        num_edges (int): Number of edges (must be at least num_users-1).
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        tuple: (edges, capacities, users, paths, G)
    """
    if seed is not None:
        random.seed(seed)
    
    # Create a connected random graph using NetworkX
    G = nx.generators.random_graphs.dense_gnm_random_graph(num_users, num_edges, seed)
    
    # Ensure all nodes are labeled as 'a0', 'a1', etc.
    mapping = {i: f'a{i}' for i in range(num_users)}
    G = nx.relabel_nodes(G, mapping)

    # Extract users
    users = list(G.nodes)

    # Extract edges
    edges = list(G.edges)

    # Assign capacities (random between 1 and 10)
    capacities = {edge: random.randint(1, 10) for edge in edges}

    # Generate random paths (random start and end points)
    paths = {}
    for _ in range(num_users // 2):  # Generate a few random paths
        start, end = random.sample(users, 2)
        try:
            path = nx.shortest_path(G, start, end)
            paths[(start, end)] = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        except nx.NetworkXNoPath:
            continue  # Skip if no path exists (shouldn't happen in a connected graph)

    return edges, capacities, users, paths, G

def plot_and_save_graph_with_paths(G, capacities, paths, filename="graph.png"):
    """
    Plots the generated graph with labeled nodes, edges, and paths highlighted, and saves the figure.
    
    Args:
        G (networkx.Graph): The generated graph.
        capacities (dict): Dictionary with edge capacities.
        paths (dict): Dictionary with different paths.
        filename (str): The name of the file to save the graph image.
    """
    plt.figure(figsize=(8, 6))
    
    pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
    
    # Draw the base graph
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", edge_color="gray", linewidths=1, font_size=12)
    
    # Draw edge labels (capacities)
    edge_labels = {edge: str(capacities[edge]) for edge in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.5)
    
    # Count how many times each edge appears in paths
    edge_usage = {}
    for path in paths.values():
        for edge in path:
            edge_usage[edge] = edge_usage.get(edge, 0) + 1

    # Highlight paths with different colors and adjust edge width
    colors = ["red", "blue", "green", "purple", "orange"]
    for i, (key, path) in enumerate(paths.items()):
        path_edges = path  # Get list of edges in the path
        color = colors[i % len(colors)]  # Cycle through colors
        
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=2.5, style="dashed", label=f"Path {key}")

    # Adjust edge width for overlapping paths
    for edge, count in edge_usage.items():
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=1.5 + count, edge_color="black")

    plt.title("Generated Random Graph with Highlighted Paths")
    plt.legend(loc="best")
    
    # Save the plot
    plt.savefig(filename, format="png")
    plt.show()

# Generate graph and plot with paths, then save
edges, capacities, users, paths, G = generate_random_graph(num_users=6, num_edges=8, seed=42)
plot_and_save_graph_with_paths(G, capacities, paths, filename=filename)

main(edges, capacities, users, paths)



