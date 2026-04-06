from langchain_core.tools import tool
from typing import Set, Tuple, Dict, List, Any



def make_retrieve_tool(gdb: "GraphDBManager"):
    @tool
    def retrieve_partial_graphs_tool(query: str, previous_groups: list, previous_diseases: list) -> tuple:
        """
        Calls GraphDBManager.retrieve_partial_graphs(query).
        Returns:
          (
            [
              {"score": float, "partial_graph": <nx graph>, "complete_graph": <nx graph>},
              ...
            ],
            flat_ranked,
            phenotype_texts,
            disease_in_nx_pairs,
            groups,
            previous_diseases,
            token_usage,
            time_taken,
          )
        """
        return gdb.retrieve_partial_graphs(query, previous_groups, previous_diseases)

    return retrieve_partial_graphs_tool
