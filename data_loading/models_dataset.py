from typing import List
import pandas as pd
from tqdm.auto import tqdm
import pickle
from random import shuffle
from sklearn.model_selection import StratifiedKFold
import json
import os
from data_loading.encoding import EncodingDataset
from lang2graph.archimate import ArchiMateNxG
from lang2graph.ecore import EcoreNxG
from lang2graph.common import LangGraph
from lang2graph.ontouml import OntoUMLNxG
import numpy as np
from settings import seed, datasets_dir, logger

dataset_to_metamodel = {
    'modelset': 'ecore',
    'ecore_555': 'ecore',
    'mar-ecore-github': 'ecore',
    'eamodelset': 'ea',
    'ontouml': 'ontouml',
}



class ModelDataset:
    def __init__(
        self,
        dataset_name: str,
        dataset_dir='datasets',
        save_dir='datasets/pickles',
        min_edges: int = -1,
        min_enr: float = -1,
        timeout=-1,
        preprocess_graph_text: callable = None,
        include_dummies=False
    ):
        self.name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.min_edges = min_edges
        self.min_enr = min_enr
        self.timeout = timeout
        self.preprocess_graph_text = preprocess_graph_text
        self.include_dummies = include_dummies

        self.graphs: List[LangGraph] = []


    def get_train_test_split(self, train_size=0.8):
        n = len(self.graphs)
        train_size = int(n * train_size)
        idx = list(range(n))
        shuffle(idx)
        train_idx = idx[:train_size]
        test_idx = idx[train_size:]
        return train_idx, test_idx
    

    def k_fold_split(
            self,  
            k=10
        ):
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        n = len(self.graphs)
        for train_idx, test_idx in kfold.split(np.zeros(n), np.zeros(n)):
            yield train_idx, test_idx


    @property
    def data(self):
        X, y = [], []
        for g in self.graphs:
            X.append(g.text)
            y.append(g.label)
        
        if self.preprocess_graph_text:
            X = [self.preprocess_graph_text(x) for x in X]
        return X, y
    
    def __get_lm_data(self, train_idx, test_idx, tokenizer, remove_duplicates=False):
        X, y = self.data
        y_enc = {label: i for i, label in enumerate(set(y))}
        y = [y_enc[label] for label in y]
        X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
        X_test, y_test = [X[i] for i in test_idx], [y[i] for i in test_idx]
        train_dataset = EncodingDataset(tokenizer, X_train, y_train, remove_duplicates=remove_duplicates)
        test_dataset = EncodingDataset(tokenizer, X_test, y_test, remove_duplicates=remove_duplicates)
        num_classes = len(set(y))
        return {
            'train': train_dataset,
            'test': test_dataset,
            'num_classes': num_classes
        }

    def get_graph_classification_data(self, tokenizer, remove_duplicates=False):
        train_idx, test_idx = self.get_train_test_split()
        return self.__get_lm_data(train_idx, test_idx, tokenizer, remove_duplicates=remove_duplicates)
    
    def get_graph_classification_data_kfold(self, tokenizer, k=10, remove_duplicates=False):
        for train_idx, test_idx in self.k_fold_split(k=k):
            yield self.__get_lm_data(train_idx, test_idx, tokenizer, remove_duplicates=remove_duplicates)


    def __repr__(self):
        return f'Dataset({self.name}, graphs={len(self.graphs)})'
    
    def __getitem__(self, key) -> LangGraph:
        return self.graphs[key]
    
    def __iter__(self):
        return iter(self.graphs)
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        print(f'Saving {self.name} to pickle')
        pkl_file = f'{self.name}{"_with_dummies" if self.include_dummies else ''}.pkl'
        with open(os.path.join(self.save_dir, pkl_file), 'wb') as f:
            pickle.dump(self.graphs, f)
        print(f'Saved {self.name} to pickle')


    def filter_graphs(self):
        # print("Filtering graphs with min edges and min enr: ", self.min_edges, self.min_enr)
        graphs = list()
        for graph in self.graphs:
            addable = True
            if self.min_edges > 0 and graph.number_of_edges() < self.min_edges:
                addable = False
            if self.min_enr > 0 and graph.enr < self.min_enr:
                addable = False
            
            if addable:
                # print("Addable because min edges and min enr: ", graph.number_of_edges())
                graphs.append(graph)
        
        self.graphs = graphs



    def load(self):
        print(f'Loading {self.name} from pickle')
        pkl_file = f'{self.name}{"_with_dummies" if self.include_dummies else ''}.pkl'
        with open(os.path.join(self.save_dir, pkl_file), 'rb') as f:
            self.graphs = pickle.load(f)
        
        self.filter_graphs()
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')
    

    @property
    def summary(self):
        num_graphs = len(self.graphs)
        num_edges = sum([g.number_of_edges() for g in self.graphs])
        num_nodes = sum([g.number_of_nodes() for g in self.graphs])
        average_nodes = num_nodes / num_graphs
        average_edges = num_edges / num_graphs
        average_n2e_ratio = np.mean([g.number_of_nodes() / g.number_of_edges() for g in self.graphs])
        return {
            'num_graphs': num_graphs,
            'num_edges': num_edges,
            'num_nodes': num_nodes,
            'average_nodes': f"{average_nodes:.2f}",
            'average_edges': f"{average_edges:.2f}",
            'average_n2e_ratio': f"{average_n2e_ratio:.2f}"
        }


class EcoreDataset(ModelDataset):
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir=datasets_dir,
            save_dir='datasets/pickles',
            reload=False,
            remove_duplicates=False,
            min_edges: int = -1,
            min_enr: float = -1,
            preprocess_graph_text: callable = None,
            include_dummies=False
        ):
        
        if include_dummies:
            assert dataset_name in ['modelset'], "Dummies are only available for modelset"
        
        super().__init__(
            dataset_name, 
            dataset_dir=dataset_dir, 
            save_dir=save_dir, 
            min_edges=min_edges, 
            min_enr=min_enr,
            preprocess_graph_text=preprocess_graph_text,
            include_dummies=include_dummies
        )
        os.makedirs(save_dir, exist_ok=True)
        data_path = os.path.join(dataset_dir, dataset_name)

        pkl_file = f'{self.name}{"_with_dummies" if self.include_dummies else ''}.pkl'
        file_name = os.path.join(data_path, 'ecore.jsonl') if not include_dummies\
            else os.path.join(data_path, 'ecore-with-dummy.jsonl')

        dataset_exists = os.path.exists(os.path.join(save_dir, pkl_file))
        print(f"Dataset exists: {dataset_exists}, reload: {reload}")
        if reload or not dataset_exists:

            
            self.graphs: List[EcoreNxG] = []
            data_path = os.path.join(dataset_dir, dataset_name)
            file_name = os.path.join(data_path, 'ecore.jsonl') if not include_dummies\
                else os.path.join(data_path, 'ecore-with-dummy.jsonl')
            
            # for file in os.listdir(data_path):
            #     if file.endswith('.jsonl') and file.startswith("ecore"):
            json_objects = json.load(open(file_name))
            for g in tqdm(json_objects, desc=f'Loading {dataset_name.title()}'):
                
                if remove_duplicates and g['is_duplicated']:
                    continue
                
                if not include_dummies and g['labels'] == 'dummy':
                    print(f"Skipping dummy graph {g['ids']}")
                    continue
                
                nxg = EcoreNxG(g)
                self.graphs.append(nxg)

            print(f'Loaded Total {self.name} with {len(self.graphs)} graphs')
            print("Filtering...")
            self.save()
            self.filter_graphs()
        else:
            self.load()
        
        logger.info(f'Loaded {self.name} with {len(self.graphs)} graphs')
        
        # if remove_duplicates:
        #     self.dedup()

        logger.info(f'Graphs: {len(self.graphs)}')
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')


    def dedup(self) -> List[EcoreNxG]:
        logger.info(f'Deduplicating {self.name}')
        return [g for g in self.graphs if not g.is_duplicated]

    def __repr__(self):
        return f"EcoreDataset({self.name}, graphs={len(self.graphs)})"


class ArchiMateDataset(ModelDataset):
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir=datasets_dir,
            save_dir='datasets/pickles',
            reload=False,
            remove_duplicates=False,
            min_edges: int = -1,
            min_enr: float = -1,
            timeout=-1,
            language=None,
            preprocess_graph_text: callable = None,
            include_dummies=False
        ):
        super().__init__(
            dataset_name, 
            dataset_dir=dataset_dir, 
            save_dir=save_dir, 
            min_edges=min_edges, 
            min_enr=min_enr,
            timeout=timeout,
            preprocess_graph_text=preprocess_graph_text,
            include_dummies=include_dummies
        )
        os.makedirs(save_dir, exist_ok=True)
        
        dataset_exists = os.path.exists(os.path.join(save_dir, f'{dataset_name}.pkl'))
        if reload or not dataset_exists:
            self.graphs: List[ArchiMateNxG] = []
            data_path = os.path.join(dataset_dir, dataset_name, 'processed-models')
            if language:
                df = pd.read_csv(os.path.join(dataset_dir, dataset_name, f'{language}-metadata.csv'))
                model_dirs = df['ID'].to_list()
            else:
                model_dirs = os.listdir(data_path)

            for model_dir in tqdm(model_dirs, desc=f'Loading {dataset_name.title()}'):
                model_dir = os.path.join(data_path, model_dir)
                if os.path.isdir(model_dir):
                    model_file = os.path.join(model_dir, 'model.json')
                    if os.path.exists(model_file):
                        model = json.load(open(model_file))
                        try:
                            nxg = ArchiMateNxG(
                                model, 
                                path=model_file,
                                timeout=timeout
                            )
                            if nxg.number_of_edges() < 1:
                                continue
                            self.graphs.append(nxg)
                            
                        except Exception as e:
                            raise e
            print("Total graphs:", len(self.graphs)) 
            self.filter_graphs()
            self.save()
        else:
            self.load()
        
        if remove_duplicates:
            self.dedup()
        
        assert all([g.number_of_edges() >= min_edges for g in self.graphs]), f"Filtered out graphs with less than {min_edges} edges"
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')
        print(f'Graphs: {len(self.graphs)}')
    

    def dedup(self) -> List[ArchiMateNxG]:
        return list({str(g.edges(data=True)): g for g in self.graphs}.values())
    
    def __repr__(self):
        return f"ArchiMateDataset({self.name}, graphs={len(self.graphs)})"


class OntoUMLDataset(ModelDataset):
    def __init__(
            self, 
            dataset_name: str, 
            dataset_dir=datasets_dir,
            save_dir='datasets/pickles',
            reload=False,
            remove_duplicates=False,
            min_edges: int = -1,
            min_enr: float = -1,
            timeout=-1,
            preprocess_graph_text: callable = None,
            include_dummies=False
        ):
        super().__init__(
            dataset_name, 
            dataset_dir=dataset_dir, 
            save_dir=save_dir, 
            min_edges=min_edges, 
            min_enr=min_enr,
            timeout=timeout,
            preprocess_graph_text=preprocess_graph_text,
            include_dummies=include_dummies
        )
        os.makedirs(save_dir, exist_ok=True)
        
        dataset_exists = os.path.exists(os.path.join(save_dir, f'{dataset_name}.pkl'))
        if reload or not dataset_exists:
            self.graphs: List[OntoUMLNxG] = []
            data_path = os.path.join(dataset_dir, dataset_name, 'models')
            model_dirs = os.listdir(data_path)

            for model_dir in tqdm(model_dirs, desc=f'Loading {dataset_name.title()}'):
                model_dir = os.path.join(data_path, model_dir)
                if os.path.isdir(model_dir):
                    model_file = os.path.join(model_dir, 'ontology.json')
                    if os.path.exists(model_file):
                        with open(model_file, encoding='iso-8859-1') as f:
                            model = json.load(f)
                        try:
                            nxg = OntoUMLNxG(model)
                            if nxg.number_of_edges() < 1:
                                continue
                            self.graphs.append(nxg)
                            
                        except Exception as e:
                            print(f"Error in {model_file} {e}")
                
            self.filter_graphs()
            self.save()
        else:
            self.load()
        
        if remove_duplicates:
            self.dedup()
        
        print(f'Loaded {self.name} with {len(self.graphs)} graphs')
        print(f'Graphs: {len(self.graphs)}')
    

    def dedup(self) -> List[OntoUMLNxG]:
        return list({str(g.edges(data=True)): g for g in self.graphs}.values())
    
    def __repr__(self):
        return f"OntoUMLDataset({self.name}, graphs={len(self.graphs)})"



def get_metamodel_dataset_type(dataset):
    return dataset_to_metamodel[dataset]


def get_model_dataset_class(dataset_name):
    dataset_type = get_metamodel_dataset_type(dataset_name)
    if dataset_type == 'ea':
        dataset_class = ArchiMateDataset
    elif dataset_type == 'ecore':
        dataset_class = EcoreDataset
    elif dataset_type == 'ontouml':
        dataset_class = OntoUMLDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return dataset_class


def get_models_dataset(dataset_name, **config_params):
    dataset_type = get_metamodel_dataset_type(dataset_name)
    if dataset_type != 'ea' and 'language' in config_params:
        del config_params['language']
    dataset_class = get_model_dataset_class(dataset_name)
    return dataset_class(dataset_name, **config_params)

