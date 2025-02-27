# 3rd Party
import graphviz
# Local
from .graph_sample import GraphSample


def to_graphviz(sample):
    G = graphviz.Digraph()
    i = 0
    for token in sample.target:
        G.node(str(i), token)
        i += 1
    for token in sample.context:
        G.node(str(i), token)
        i += 1
    for concept in sample.kb:
        G.node(str(i), concept)
        i += 1 
    for edge in sample.edges:
        G.edge(str(edge.head_node_index), str(edge.tail_node_index))#, constraint='false')
    return G

def neighborhood(sample: GraphSample, label, subset = 'concept'):
    G = graphviz.Digraph()
    if subset == 'concept':
        index = len(sample.target) + len(sample.context) + sample.kb.index(label)
    elif subset == 'context':
        index = len(sample.target) + sample.context.index(label)
    else:
        assert subset == 'target'
        index = sample.target.index(label)
    neighborhood = set()
    kept_edges = []
    for edge in filter(lambda e: e.head_node_index == index or e.tail_node_index == index, sample.edges):
        neighborhood.add(edge.head_node_index)
        neighborhood.add(edge.tail_node_index)
        kept_edges.append(edge)
    for (i, label) in filter(lambda trip: trip[0] in neighborhood, enumerate(sample.target + sample.context + sample.kb)):
        G.node(str(i), label)
    for edge in kept_edges:
        G.edge(str(edge.head_node_index), str(edge.tail_node_index))#, constraint='false')
        pass
    return G