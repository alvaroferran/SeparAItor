from kfold_dataset import refresh_k_fold_dataset

src = "dataIn"
dest = "dataOut"
nb_folds = 4

for i in range(nb_folds):
    refresh_k_fold_dataset(src, dest, nb_folds)
