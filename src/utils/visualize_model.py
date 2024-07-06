import os
from torchview import draw_graph
from graphviz.graphs import Digraph
from model import VisionTransformer 

def generate_graph(model: VisionTransformer, destination: str) -> Digraph:
    """
    Creates a graph to visualize the architecture of the Vision Transformer.

    Parameters
    ----------
    model: VisionTransformer
        The model to be visualized.

    destination: str
        The path where the generated graph will be saved.

    Returns
    -------
    graoh: Digraph
        A digraph object that visualizes the model.
    """

    model_graph = draw_graph(
        model, input_size=(1, 3, 512, 512),
        graph_name="vision-transformer",
        expand_nested=True,
        save_graph=True, directory=destination,
        filename="vision-transformer-architecture"
    )

    graph = model_graph.visual_graph
    
    return graph

if __name__ == "__main__":
    destination = os.path.join("..", "..", "assets", "architecture")
    patch_size = 16
    num_channels = 3
    embed_dim = 768
    sequence_len = ((512 // patch_size) ** 2) + 1
    num_heads = 12
    hidden_size = 3072
    num_layers = 12
    model = VisionTransformer(patch_size, num_channels, embed_dim, sequence_len, num_heads, num_layers, hidden_size, 0.0, 10)

    if not os.path.isdir(destination):
        os.makedirs(destination)

    generate_graph(model, destination)