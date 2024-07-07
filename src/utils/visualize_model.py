import os
from get_model import get_model
from torchview import draw_graph
from ViT import VisionTransformer
from graphviz.graphs import Digraph

def generate_graph(model: VisionTransformer, variant: str, destination: str) -> Digraph:
    """
    Creates a graph to visualize the architecture of the Vision Transformer.

    Parameters
    ----------
    model: VisionTransformer
        The model to be visualized.

    variant: str
        The ViT variant

    destination: str
        The path where the generated graph will be saved.

    Returns
    -------
    graph: Digraph
        A digraph object that visualizes the model.
    """

    model_graph = draw_graph(
        model, input_size=(1, 3, 512, 512),
        graph_name="vision-transformer",
        expand_nested=True,
        save_graph=True, directory=destination,
        filename=f"ViT-{variant}-architecture"
    )

    graph = model_graph.visual_graph
    
    return graph

if __name__ == "__main__":
    destination = os.path.join("..", "..", "assets", "architecture")
    
    if not os.path.isdir(destination):
        os.makedirs(destination)
    
    for variant in ["base", "large", "huge"]:
        model = get_model(
            512, 16, variant=variant, dropout_probability=0.0, num_classes=10
        )

        generate_graph(model, variant, destination)