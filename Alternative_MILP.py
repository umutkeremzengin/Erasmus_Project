import gurobipy as gp
from gurobipy import GRB
import numpy as np

def optimize_truck_assignment(trucks):
    # Set of slots (hours of the day)
    L = list(range(24))
    # Time intervals (-1, 0, 1)
    T = [-1, 0, 1]

    # Probabilities for each truck being assigned to each slot in each period
    prob = {(j, l, t): trucks[j]['probabilities'][l][t + 1] for j in range(len(trucks)) for l in L for t in T}

    # Create a new model
    model = gp.Model("Truck_Assignment_Stochastic_No_Processing_Req")

    # Decision variables
    x = {}
    for j in range(len(trucks)):
        for l in range(24):
            for t in T:
                x[j, l, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{l}_{t}")

    # Objective function: Maximize the sum of probabilities
    obj = gp.quicksum(x[j, l, t] * prob[j, l, t] for j in range(len(trucks)) for l in range(24) for t in T)
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints

    # Each truck should be assigned to exactly 'V' slots in total
    for j in range(len(trucks)):
        model.addConstr(gp.quicksum(x[j, l, t] for l in range(24) for t in T) == trucks[j]['V'], name=f"truck_total_capacity_{j}")

    # Each slot can have at most one truck assigned in one period
    for l in range(24):
        for t in T:
            model.addConstr(gp.quicksum(x[j, l, t] for j in range(len(trucks))) <= 1, name=f"slot_occupancy_{l}_{t}")

    # Optimize the model
    model.optimize()

    # Output results
    if model.status == GRB.OPTIMAL:
        assignments = []
        for j in range(len(trucks)):
            for l in range(24):
                for t in T:
                    if x[j, l, t].X > 0.5:
                        assignments.append((trucks[j]['truck_id'], l, t))
        return assignments
    else:
        # If the model is infeasible, print the infeasibility report
        print("Model is infeasible.")
        model.computeIIS()
        model.write("model.ilp")
        return "No optimal solution found."

def get_truck_data():
    trucks = []
    n = int(input("Enter the number of trucks: "))
    for i in range(n):
        input_data = input(f"Enter data for truck {i+1} in the format (duration, capacity, preferred_time, prob_array, id): ")
        try:
            # Using eval with a dictionary to include np.array
            input_data = eval(input_data, {"array": np.array})
            if len(input_data) != 5:
                raise ValueError("Input data must contain exactly 5 elements.")
            duration, capacity, preferred_time, prob_array, truck_id = input_data
            trucks.append({
                'truck_id': truck_id,
                'P': duration,
                'V': capacity,
                'H': preferred_time,
                'probabilities': prob_array
            })
        except (SyntaxError, ValueError, NameError) as e:
            print(f"Invalid input format for truck {i+1}: {e}")
            return []
    return trucks

# Example input data
example_data = """
(1, 29, 18, array([[0.63333333, 0.23333333, 0.13333333],
       [0.66666667, 0.2       , 0.13333333],
       [0.66666667, 0.2       , 0.13333333],
       [0.66666667, 0.2       , 0.13333333],
       [0.66666667, 0.2       , 0.13333333],
       [0.7       , 0.16666667, 0.13333333],
       [0.73333333, 0.13333333, 0.13333333],
       [0.73333333, 0.13333333, 0.13333333],
       [0.73333333, 0.13333333, 0.13333333],
       [0.73333333, 0.13333333, 0.13333333],
       [0.7       , 0.13333333, 0.16666667],
       [0.66666667, 0.13333333, 0.2       ],
       [0.73333333, 0.13333333, 0.13333333],
       [0.73333333, 0.13333333, 0.13333333],
       [0.73333333, 0.13333333, 0.13333333],
       [0.73333333, 0.16666667, 0.1       ],
       [0.73333333, 0.16666667, 0.1       ],
       [0.8       , 0.1       , 0.1       ],
       [0.76666667, 0.1       , 0.13333333],
       [0.8       , 0.03333333, 0.16666667],
       [0.83333333, 0.03333333, 0.13333333],
       [0.8       , 0.06666667, 0.13333333],
       [0.66666667, 0.1       , 0.23333333],
       [0.63333333, 0.16666667, 0.2       ]]), '16044073679994652')
"""

trucks = get_truck_data()
if trucks:
    assignments = optimize_truck_assignment(trucks)
    if isinstance(assignments, str):
        print(assignments)
    else:
        for assignment in assignments:
            print(f"Truck {assignment[0]} is assigned to slot {assignment[1]} in period {assignment[2]}")
