# Import the required modules
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Create a directed acyclic graph (DAG)
model = BayesianNetwork([('A', 'C'), ('B', 'C')])

# Define the Conditional Probability Tables (CPTs) for each variable
cpd_a = TabularCPD(variable='A', variable_card=3, values=[[0.6], [0.1],[0.3]])
cpd_b = TabularCPD(variable='B', variable_card=3, values=[[0.2], [0.3],[0.5]])
cpd_c = TabularCPD(variable='C', variable_card=2, 
                    values=[[0.9, 0.8, 0.7, 0.1, 0.6, 0.2, 0.5, 0.4, 0],
                             [0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.5, 0.6, 1]],
                    evidence=['A', 'B'], evidence_card=[3, 3])

# Add the edges and CPTs to the DAG
model.add_cpds(cpd_a, cpd_b)

model.add_cpds(cpd_c)
# Use the Infer method to perform the queries
infer = VariableElimination(model)

print(cpd_c)

print(infer.query(['C']))

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD



# Define the first network structure
