# from data_loading.models_dataset import ArchiMateDataset, EcoreDataset
# import argparse

# parser = argparse.ArgumentParser(description='Run tests on datasets')
# parser.add_argument('--reload', action='store_true', help='Reload datasets from source')
# args = parser.parse_args()


# config_params = dict(
#     min_enr = 1.2,
#     min_edges = 10,
# )
# ecore = EcoreDataset('ecore_555', reload=args.reload, **config_params)
# modelset = EcoreDataset('modelset', reload=args.reload, remove_duplicates=True, **config_params)
# mar = EcoreDataset('mar-ecore-github', reload=args.reload, **config_params)
# eamodelset = ArchiMateDataset('eamodelset', reload=args.reload, **config_params)

from examples import run_pipeline, run_tree_pipeline

# run_pipeline.main()
run_pipeline.main()
# from examples.run_tree_pipeline import (
#     run_tree_gnn_node_classification,
#     run_tree_text_node_classification,
# )
