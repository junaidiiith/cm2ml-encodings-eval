import xmltodict
from .common import LangGraph
import json
from settings import logger



REFERENCE = 'reference'
SUPERTYPE = 'supertype'
CONTAINMENT = 'containment'

EGenericType = 'EGenericType'
EPackage = 'EPackage'
EClass = 'EClass'
EAttribute = 'EAttribute'
EReference = 'EReference'
EEnum = 'EEnum'
EEnumLiteral = 'EEnumLiteral'
EOperation = 'EOperation'
EParameter = 'EParameter'
EDataType = 'EDataType'
GenericNodes = [EGenericType, EPackage]



class EcoreNxG(LangGraph):
    def __init__(self, json_obj: dict):
        super().__init__()
        self.xmi = json_obj.get('xmi')
        self.graph_id = json_obj.get('ids')
        self.json_obj = json_obj
        self.graph_type = json_obj.get('model_type')
        self.label = json_obj.get('labels')
        self.is_duplicated = json_obj.get('is_duplicated')
        self.directed = json.loads(json_obj.get('graph')).get('directed')
        # self.text = doc_tokenizer(json_obj.get('txt'))

        self.__create_graph()
        self.set_numbered_labels()        
    
    
    def __create_graph(self):
        model = xmltodict.parse(self.xmi)
        eclassifiers, _ = get_eclassifiers(model)
        classifier_nodes = dict()
        for eclassifier in eclassifiers:
            eclassifier_info = get_eclassifier_info(eclassifier)
            classifier_nodes[eclassifier_info['name']] = eclassifier_info
        
        references = get_connections(classifier_nodes)
        
        for classifier_name, classifier_info in classifier_nodes.items():
            # if classifier_info['type'] != 'class':
            #     continue
            structural_features = classifier_info.get('structural_features', [])
            attributes = list()
            for f in structural_features:
                if f['type'] == 'ecore:EAttribute':
                    name = f['name']
                    attributes.append(name)

            self.add_node(
                classifier_name, 
                name=classifier_name, 
                attributes=attributes,
                abstract=classifier_info['abstract']
            )
            
        for edge in references:
            src, dest = edge['source'], edge['target']
            name = edge['name'] if 'name' in edge else ''
            self.add_edge(src, dest, name=name, type=edge['type'])
        
        for node in self.nodes:
            self.nodes[node]['abstract'] = self.nodes[node]['abstract'] if 'abstract' in self.nodes[node] and self.nodes[node]['abstract'] is not None else False

        logger.info(f'Graph {self.graph_id} created with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges')

    def __str__(self):
        return self.__repr__()



    def __repr__(self):
        reference_edges = [edge for edge in self.edges if self.edges[edge]['type'] == REFERENCE]
        containment_edges = [edge for edge in self.edges if self.edges[edge]['type'] == CONTAINMENT]
        supertype_edges = [edge for edge in self.edges if self.edges[edge]['type'] == SUPERTYPE]
        return f'EcoreNxG({self.graph_id}, nodes={self.number_of_nodes()}, edges={self.number_of_edges()}, references={len(reference_edges)}, containment={len(containment_edges)}, supertypes={len(supertype_edges)})'



def get_eclassifiers(json_obj):
    def get_eclassifiers_util(json_obj, classifiers: list):
        for key, value in json_obj.items():
            if key == 'eClassifiers':
                if isinstance(value, dict):
                    value = [value]
                classifiers.extend(value)
            elif isinstance(value, dict):
                get_eclassifiers_util(value, classifiers)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        get_eclassifiers_util(item, classifiers)
    classifiers = list()
    get_eclassifiers_util(json_obj, classifiers)
    names = [c['@name'] for c in classifiers]
    return classifiers, len(names) - len(set(names))


def get_connections(nodes):
    links = list()
    for source_class, classifier_info in nodes.items():
        if classifier_info['type'] != 'class':
            continue
        super_types = classifier_info['super_types']
        for super_type in super_types:
            if super_type in nodes:
                links.append({
                    'source': source_class,
                    'target': super_type,
                    'type': SUPERTYPE,
                })
                nodes[super_type]['abstract'] = True

        for feature in classifier_info['structural_features']:
            ref = feature['ref']
            if ref and ref in nodes:
                links.append({
                    'name': feature['name'],
                    'source': source_class,
                    'target': ref,
                    'type': REFERENCE if not feature['containment'] else CONTAINMENT
                })
    
    for node in nodes:
        abstract = nodes[node].get('abstract', '')
        if abstract:
            nodes[node]['abstract'] = True
        else:
            nodes[node]['abstract'] = False

    return links


def get_estructural_feature(structural_feat):
    feat_type = '@xsi:type' if '@xsi:type' in structural_feat else '@xmi:type'
    structural_feat_type = structural_feat[feat_type]
    name = structural_feat['@name']
    eType = structural_feat['@eType'] if '@eType' in structural_feat else False
    
    return {
        'name': name,
        'ref': eType.split('/')[-1] if eType else None,
        'type': structural_feat_type,
        'containment': structural_feat['@containment'] if '@containment' in structural_feat else None,
    }


def get_eclassifier_info_eclass(eclass):
    name = eclass['@name']
    super_types = eclass['@eSuperTypes'] if '@eSuperTypes' in eclass else ""
    super_types = [s.split('/')[-1] for s in super_types.split(' ')] if super_types else []
    structural_features = eclass['eStructuralFeatures'] if 'eStructuralFeatures' in eclass else []
    if not isinstance(structural_features, list):
        structural_features = [structural_features]
    
    structural_features_info = list()
    for feature in structural_features:
        structural_features_info.append(get_estructural_feature(feature))
    
    return {
        'name': name,
        'type': 'class',
        'super_types': super_types,
        'structural_features': structural_features_info,
        'abstract': '@abstract' in eclass and eclass['@abstract']
    }
        
def get_eclassifier_info_eenum(eenum):
    name = eenum['@name']
    literals = eenum['eLiterals'] if 'eLiterals' in eenum else []
    if not isinstance(literals, list):
        literals = [literals]
    
    literals_info = list()
    for literal in literals:
        literal_label = '@literal' if '@literal' in literal else '@value'
        name = literal['@name']
        value = literal[literal_label] if literal_label in literal else ""
        literals_info.append((name, value))
    
    return {
        'name': name,
        'type': 'enum',
        'literals': literals_info
    }

def get_eclassifier_info_edatatype(edatatype):
    name = edatatype['@name']
    return {
        'type': 'datatype',
        'name': name,
    }


def get_eclassifier_info(eclassifier):
    classifier_type = '@xsi:type' if '@xsi:type' in eclassifier else '@xmi:type'
    if classifier_type not in eclassifier:
        raise ValueError(f"Classifier has no type: {eclassifier}")
    if eclassifier[classifier_type] in ['ecore:EClass', 'EClass']:
        return get_eclassifier_info_eclass(eclassifier)
    elif eclassifier[classifier_type] in ['ecore:EEnum', 'EEnum']:
        return get_eclassifier_info_eenum(eclassifier)
    elif eclassifier[classifier_type] in ['ecore:EDataType', 'EDataType']:
        return get_eclassifier_info_edatatype(eclassifier)
    else:
        logger.log(eclassifier)
        raise ValueError(f"Unknown classifier type: {eclassifier[classifier_type]}")