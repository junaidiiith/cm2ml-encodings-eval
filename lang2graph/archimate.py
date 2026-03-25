from .common import LangGraph


class ArchiMateNxG(LangGraph):
    def __init__(
        self, 
        json_obj: dict, 
        path: str,
        timeout = -1
    ):
        super().__init__()
        self.json_obj = json_obj
        self.timeout = timeout
        self.path = path
        self.graph_id = json_obj['identifier'].split('/')[-1]


        self.__create_graph()
        self.set_numbered_labels()


    def __create_graph(self):
        for node in self.json_obj['elements']:
            self.add_node(node['id'], **node)
        for edge in self.json_obj['relationships']:
            self.add_edge(edge['sourceId'], edge['targetId'], **edge)
