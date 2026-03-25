import json
from tqdm.auto import tqdm
from lang2graph.common import LangGraph
from utils import find_files_with_extension
from settings import logger


ONTOUML_ELEMENT_ID = 'id'
ONTOUML_ELEMENT_TYPE = 'type'
ONTOUML_ELEMENT_NAME = 'name'
ONTOUML_ELEMENT_DESCRIPTION = 'description'

ONTOUML_GENERALIZATION = "Generalization"
ONTOUML_GENERALIZATION_GENERAL = "general"
ONTOUML_GENERALIZATION_SPECIFIC = "specific"
ONTOUML_GENERALIZATION_SET = "GeneralizationSet"
ONTOUML_GENERALIZATION_SET_GENERALIZATIONS = "generalizations"
ONTOUML_GENERALIZATION_SET_IS_DISJOINT = "isDisjoint"
ONTOUML_GENERALIZATION_SET_IS_COMPLETE = "isComplete"

ONTOUML_PROJECT = "Project"
ONTOUML_PROJECT_MODEL = "model"
ONTOUML_PROJECT_MODEL_CONTENTS = "contents"
ONTOUML_RELATION = "Relation"
ONTOUML_PROPERTIES = "properties"
ONTOUML_RELATION_PROPERTY_TYPE = "propertyType"
ONTOUML_STEREOTYPE = "stereotype"
ONTOUML_CLASS = "Class"
ONTOUML_ENUMERATION = "enumeration"
ONTOUML_CLASS_LITERALS = 'literals'
ONTOUML_PACKAGE = "Package"
ONTOUML_LITERAL = "Literal"


extra_properties = [
    "isAbstract", 
    "isDerived", 
    "isDisjoint", 
    "type", 
    "isComplete", 
    "isPowertype", 
    "isExtensional", 
    "isOrdered",
    "aggregationKind",
]


class OntoUMLNxG(LangGraph):
    def __init__(self, json_obj: dict, rel_as_node=True):
        super().__init__()
        self.graph_id = json_obj['id']
        self.json_obj = json_obj
        self.rel_as_node = rel_as_node
        self.__create_graph()
        self.set_numbered_labels()

        self.text = " ".join([
            self.nodes[node]['name'] if 'name' in self.nodes[node] and self.nodes[node]['name'] else ''
            for node in self.nodes
        ])
    
    def __create_graph(self):
        
        def ontouml_id2obj(obj):
            assert isinstance(obj, dict)
            for key in obj:
                if key == ONTOUML_ELEMENT_ID and ONTOUML_ELEMENT_TYPE in obj and obj[ONTOUML_ELEMENT_TYPE]\
                    in [ONTOUML_CLASS, ONTOUML_RELATION, ONTOUML_GENERALIZATION_SET, ONTOUML_GENERALIZATION]\
                        and ONTOUML_ELEMENT_DESCRIPTION in obj:
                    id2obj_map[obj[ONTOUML_ELEMENT_ID]] = obj
                elif isinstance(obj[key], dict):
                    ontouml_id2obj(obj[key])
                elif isinstance(obj[key], list):
                    for item in obj[key]:
                        assert not isinstance(item, list)
                        if isinstance(item, dict):
                            ontouml_id2obj(item)

        def create_nxg():

            for k, v in id2obj_map.items():
                node_name = v.get('name', '')
                
                if v[ONTOUML_ELEMENT_TYPE] in [ONTOUML_CLASS, ONTOUML_RELATION]:
                    self.add_node(k, name=node_name, type=v[ONTOUML_ELEMENT_TYPE], description='')
                    for prop in extra_properties:
                        self.nodes[k][prop] = v[prop] if prop in v else False

                    logger.info(f"Node: {node_name} type: {v[ONTOUML_ELEMENT_TYPE]}")
                # else:
                #     continue
                
                logger.info(f"Node: {node_name} type: {v[ONTOUML_ELEMENT_TYPE]}")
                if ONTOUML_STEREOTYPE in v and v[ONTOUML_STEREOTYPE] is not None:
                    self.nodes[k][ONTOUML_STEREOTYPE] = v[ONTOUML_STEREOTYPE].lower()
                    logger.info(f"Stereotype: {v[ONTOUML_STEREOTYPE].lower()}")
                

                if ONTOUML_ELEMENT_DESCRIPTION in v and v[ONTOUML_ELEMENT_DESCRIPTION] is not None:
                    self.nodes[k][ONTOUML_ELEMENT_DESCRIPTION] = v[ONTOUML_ELEMENT_DESCRIPTION]
                    logger.info(f"Description: {v[ONTOUML_ELEMENT_DESCRIPTION]}")
                

                if v[ONTOUML_ELEMENT_TYPE] == ONTOUML_CLASS:
                    if ONTOUML_CLASS_LITERALS in v and v[ONTOUML_CLASS_LITERALS] is not None:
                        literals = v[ONTOUML_CLASS_LITERALS] if isinstance(v[ONTOUML_CLASS_LITERALS], list) else [v[ONTOUML_CLASS_LITERALS]]
                        literals_str = ", ".join([literal[ONTOUML_ELEMENT_NAME] for literal in literals])
                        self.nodes[k][ONTOUML_PROPERTIES] = literals_str

                        logger.info(f"Literals: {literals_str}")
                    
                    elif ONTOUML_PROPERTIES in v and v[ONTOUML_PROPERTIES] is not None:
                        properties = v[ONTOUML_PROPERTIES] if isinstance(v[ONTOUML_PROPERTIES], list) else [v[ONTOUML_PROPERTIES]]
                        self.nodes[k][ONTOUML_PROPERTIES] = [property[ONTOUML_ELEMENT_NAME] for property in properties]
                        

                elif v[ONTOUML_ELEMENT_TYPE] == ONTOUML_RELATION:    
                    properties = v[ONTOUML_PROPERTIES] if isinstance(v[ONTOUML_PROPERTIES], list) else [v[ONTOUML_PROPERTIES]]
                    assert len(properties) == 2
                    try:
                        x_id = properties[0][ONTOUML_RELATION_PROPERTY_TYPE][ONTOUML_ELEMENT_ID]
                        y_id = properties[1][ONTOUML_RELATION_PROPERTY_TYPE][ONTOUML_ELEMENT_ID]
                        x_name = id2obj_map[x_id][ONTOUML_ELEMENT_NAME] if ONTOUML_ELEMENT_NAME is not None else ''
                        y_name = id2obj_map[y_id][ONTOUML_ELEMENT_NAME] if ONTOUML_ELEMENT_NAME is not None else ''

                        self.add_edge(x_id, v[ONTOUML_ELEMENT_ID], type='rel')
                        self.add_edge(v[ONTOUML_ELEMENT_ID], y_id, type='rel')

                        logger.info(f"\tRelationship:, {x_name} --> {y_name}\n")
                    except TypeError as e:
                        # print(f"Error in {v[ONTOUML_ELEMENT_TYPE]}, {v[ONTOUML_ELEMENT_NAME]}")
                        pass

                
                elif v[ONTOUML_ELEMENT_TYPE] == ONTOUML_GENERALIZATION:
                    general = v[ONTOUML_GENERALIZATION_GENERAL][ONTOUML_ELEMENT_ID]
                    specific = v[ONTOUML_GENERALIZATION_SPECIFIC][ONTOUML_ELEMENT_ID]
                    general_name = id2obj_map[general][ONTOUML_ELEMENT_NAME]\
                        if ONTOUML_ELEMENT_NAME in id2obj_map[general] else ''
                    specific_name = id2obj_map[specific][ONTOUML_ELEMENT_NAME] \
                        if ONTOUML_ELEMENT_NAME in id2obj_map[specific] else ''

                    logger.info(f"\tGeneralization:, {specific_name} -->> {general_name}\n")
                    self.add_edge(specific, general, type='gen')

        def create_nxg_rel_as_edge():
            # TODO: To be implemented
            raise NotImplementedError


        id2obj_map = dict()
        ontouml_id2obj(self.json_obj)
        if self.rel_as_node:
            create_nxg()
        else:
            create_nxg_rel_as_edge()


def get_ontouml_to_nx(data_dir, min_stereotypes=10):
    ontouml_graphs = list()
    models = find_files_with_extension(data_dir, "json")
    for mfp in tqdm(models, desc=f"Reading {len(models)} OntoUML models"):
        if mfp.endswith(".ecore") or mfp.endswith(".json"):
            json_obj = json.loads(open(mfp, 'r', encoding='iso-8859-1').read())
            g = OntoUMLNxG(json_obj)
            stereotype_nodes = [node for node, stereotype in g.nodes(data=ONTOUML_STEREOTYPE) if stereotype is not None]
            if len(stereotype_nodes) >= min_stereotypes:
                ontouml_graphs.append((g, mfp))
    
    return ontouml_graphs
