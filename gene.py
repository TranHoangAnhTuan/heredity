from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from heredity import PROBS
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

# Define the hereditary distribution of impairement hearing
hereditary_distribution = [
    [0.0001, 0.0051, 0.9802, 0.0051, 0.2601, 0.5049, 0.9802, 0.5049, 0.9801],
    [0.0198, 0.5098, 0.0099, 0.5098, 0.4998, 0.4902, 0.0099, 0.4902, 0.0198], 
    [0.9801, 0.4851, 0.0099, 0.4851, 0.2401, 0.0049, 0.0099, 0.0049, 0.0001]
]
def initalize_model():

    model = BayesianNetwork()
    return model

def create_model(person, model):
    """
    Return a model with person variable as string name
    e.g.: person = "Harry" then we combine name with Gene or Trait "HarryGene" and "HarryTrait"
    
    will return Bayes network with ('HarryGene', 'HarryTrait')
    with Gene and Trait of one person are above PROBS
    """
    gene_cpd = TabularCPD(variable='{}Gene'.format(person),
                          variable_card=3, 
                          values=[[PROBS["gene"][0]], [PROBS["gene"][1]], [PROBS["gene"][2]]],
                          state_names={'{}Gene'.format(person): [0,1,2]})

    trait_cpd = TabularCPD(variable='{}Trait'.format(person),
                           variable_card=2,
                           values=[
                             
                               [PROBS["trait"][0][False], PROBS["trait"][1][False], PROBS["trait"][2][False]],
                                 [PROBS["trait"][0][True], PROBS["trait"][1][True], PROBS["trait"][2][True]]
                           ],
                           evidence=['{}Gene'.format(person)],
                           evidence_card=[3])
    model2 = BayesianNetwork([('{}Gene'.format(person),'{}Trait'.format(person))])
    model2.add_cpds(gene_cpd, trait_cpd)
    combine_model(model, model2)


def add_relation(child, mother, father, model):
    """
    we have name of parent and name of child then we set relation of inheritance base on
    above hereditary_distribution
    e.g : James is father and Lily is mother of Harry then we set ('JamesGene','HarryGene') 
    and ('LilyGene','Harry') to current model just input
    note: evidence card is [3,3] evidence is [father+'Gene',mother + 'Gene'] and
    values equal hereditary_distribution
    gene have 3 
    """
    # Make a copy of existing model

    # Add edges from parent genes to child gene
    model.add_edges_from([(mother + 'Gene', child + 'Gene'), 
                        (father + 'Gene', child + 'Gene'),
                        (child + 'Gene', child + 'Trait')])

    # Create a new tabular CPD for the child gene
    child_gene_cpd = TabularCPD(variable=child + 'Gene',
                                variable_card=3,
                                values=hereditary_distribution,
                                evidence=[mother + 'Gene', father + 'Gene'],
                                evidence_card=[3, 3])
  
    trait_cpd = TabularCPD(variable='{}Trait'.format(child),
                           variable_card=2,
                           values=[
                               [PROBS["trait"][0][False], PROBS["trait"][1][False], PROBS["trait"][2][False]],
                                [PROBS["trait"][0][True], PROBS["trait"][1][True], PROBS["trait"][2][True]]
                           ],
                           evidence=[child + 'Gene'],
                           evidence_card=[3])

    # Add the new CPD to the updated model
    model.add_cpds(child_gene_cpd,trait_cpd)


    return model

    # add the new CPD to the second network


        
               

def combine_model(curModel, sumodelodel):
    """
    Combine sumodelodel into curModel using BayesianNetwork
    """
    curModel.add_nodes_from(sumodelodel.nodes())
    curModel.add_edges_from(sumodelodel.edges())
    curModel.add_cpds(*sumodelodel.get_cpds())
    
    return curModel

from pgmpy.models import BayesianModel

def is_exist(father, mother, model):
    """
    mother and father Node have format 
    (father + "Gene", father + "Trait") and mother is the same 
    """
    # Create a new BayesianModel object from the given model.
    
    # Check if both father and mother exist in the model as nodes.
    if father+"Gene" in model.nodes() and father+"Trait" in model.nodes() and \
       mother+"Gene" in model.nodes() and mother+"Trait" in model.nodes():
        return True
    else:
        return False


def inference_by_evidences(people , model):
    evidences = {}
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }
    
    infer = VariableElimination(model)
    # print(evidences)
    # update gene probabilities
    for person in people:
        # for i in range(3):
        #     if people[person]['gene'][i] is not None:
        #         evidences[person + "Gene"] = people[person]['gene'][i]
        
        if people[person]['trait']:
            evidences[person + "Trait"] = 1
            probabilities[person]['trait'][False] = 0
            probabilities[person]['trait'][True] = 1

        elif people[person]['trait'] is not None:
            evidences[person + "Trait"] = 0
            probabilities[person]['trait'][False] = 1
            probabilities[person]['trait'][True] = 0


    for person in people:
        gene_dist = infer.query(variables=[person + "Gene"], evidence=evidences)
        for index, prob in enumerate(gene_dist.values):
            # value = gene_dist.variables[0].state_names[index]
            probabilities[person]["gene"][index] = prob
            # print(index, prob)

    print(evidences)
    # update trait probabilities
    for person in people:
        if person + "Trait" in evidences:
            continue



        else:
            trait_dist = infer.query([person + "Trait"], evidence=evidences)
            for index, prob in enumerate(trait_dist.values):
                # value = trait_dist.variables[0].state_names[index]
                probabilities[person]["trait"][index] = prob
                print(index, prob)


    return probabilities


def draw_model(model):
    nx.draw_networkx(model, pos=nx.drawing.nx_pydot.graphviz_layout(model, prog='dot'),
                node_color='lightblue', edge_color='gray', font_size=12, font_weight='bold',
                with_labels=True, arrows=True)
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

def draw_bayesian_model(model):
    # Create a directed graph object using networkx
    nx_model = nx.DiGraph()

    # Add nodes to the graph
    for node in model.nodes():
        nx_model.add_node(node)

    # Add edges to the graph
    for edge in model.edges():
        nx_model.add_edge(edge[0], edge[1])

    # Set node positions using the graphviz_layout function from networkx
    pos = nx.drawing.nx_pydot.graphviz_layout(nx_model, prog='dot')

    # Set node colors based on their type (observed or hidden)
    node_colors = []
    for node in model.nodes():
        if node.startswith('O'):
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightblue')

    # Set edge colors
    edge_colors = 'gray'

    # Set font size and font weight
    font_size = 12
    font_weight = 'bold'

    # Create a plot object
    plt.figure(figsize=(8, 6))

    # Draw the graph using networkx
    nx.draw(nx_model, pos=pos, node_color=node_colors, edge_color=edge_colors,
            font_size=font_size, font_weight=font_weight, with_labels=True, arrows=True)

    # Add node probabilities to the graph using pgmpy's TabularCPD
    for node in model.nodes():
        if node.startswith('O'):
            # If the node is an observed variable, get its CPD and create a table
            cpd = model.get_cpds(node)
            values = cpd.get_values().flatten()
            table = []
            for i in range(0, len(values), 2):
                table.append([round(values[i], 2), round(values[i+1], 2)])
            table.reverse()
            # Add the table to the graph using the add_table function from networkx
            nx.draw_networkx_labels(nx_model, pos, labels={node: table}, font_size=font_size, font_weight=font_weight)

    # Show the plot
    plt.show()

