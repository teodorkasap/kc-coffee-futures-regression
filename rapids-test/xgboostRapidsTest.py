from dask.distributed import Client
from dask_cuda import LocalCUDACluster
# from dask import dataframe as dd
import xgboost as xgb
import dask_cudf


def main(client):
    fname = 'HIGGS.csv'
    colnames = ["label"] + ["feature-%02d" % i for i in range(1, 29)]
    dask_df = dask_cudf.read_csv(fname, header=None, names=colnames)
    y = dask_df["label"]
    X = dask_df[dask_df.columns.difference(["label"])]
    dtrain = xgb.dask.DaskDMatrix(client, X, y)

    output = xgb.dask.train(client,
                            {'tree_method': 'gpu_hist'},
                            dtrain,
                            num_boost_round=100,
                            evals=[(dtrain, 'train')])
    booster = output['booster']
    history = output['history']  
    booster.save_model('xgboost-model')
    print('Training evaluation history:', history)


if __name__ == '__main__':
    with LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster) as client:
            main(client)
