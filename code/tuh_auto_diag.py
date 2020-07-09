from sklearn.model_selection import KFold
import torch.nn.functional as F
import pandas as pd
import torch as th
import numpy as np
import logging
import json
import time
import sys
import os

from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from braindecode.torch_ext.util import np_to_var, var_to_np, set_random_seeds
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.optimizers import AdamW
from braindecode.models.eegnet import EEGNetv4
from braindecode.models.deep4 import Deep4Net

sys.path.insert(1, "/home/gemeinl/code/NeuralArchitectureSearch/")
from src.deep_learning.pytorch.models.tcn_model import TemporalConvNet
from src.deep_learning.pytorch.optimizer import ExtendedAdam

sys.path.insert(1, "/home/gemeinl/code/braindecode_lazy/")
from braindecode_lazy.experiments.monitors_lazy_loading import (
    LazyMisclassMonitor, RMSEMonitor, CroppedDiagnosisMonitor, RAMMonitor,
    compute_preds_per_trial)
from braindecode_lazy.datautil.iterators import LazyCropsFromTrialsIterator
from braindecode_lazy.datasets.tuh_lazy import TuhLazy, TuhLazySubset
from braindecode_lazy.datasets.tuh import Tuh, TuhSubset
from examples.utils import parse_run_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("test")


def nll_loss_on_mean(preds, targets):
    mean_preds = th.mean(preds, dim=2, keepdim=False)
    return F.nll_loss(mean_preds, targets)


def remove_file_from_dataset(dataset, file_id, file):
    if file in dataset.file_paths[file_id]:
        indices = list(range(len(dataset)))
        indices.remove(file_id)

        ds = TuhSubset(dataset, indices)
        for f in ds.file_paths:
            assert file not in f, "file {} not correctly removed".format(f)
        logging.info("successfully removed rec {}".format(f))
        return ds
    else:
        return dataset


def setup_exp(
        train_folder,
        n_recordings,
        n_chans,
        model_name,
        n_start_chans,
        n_chan_factor,
        input_time_length,
        final_conv_length,
        model_constraint,
        stride_before_pool,
        init_lr,
        batch_size,
        max_epochs,
        cuda,
        num_workers,
        task,
        weight_decay,
        n_folds,
        shuffle_folds,
        lazy_loading,
        eval_folder,
        result_folder,
        run_on_normals,
        run_on_abnormals,
        seed,
        l2_decay,
        gradient_clip,
        ):
    info_msg = "using {}, {}".format(
        os.environ["SLURM_JOB_PARTITION"], os.environ["SLURMD_NODENAME"],)
    info_msg += ", gpu {}".format(os.environ["CUDA_VISIBLE_DEVICES"])
    logging.info(info_msg)

    logging.info("Targets for this task: <{}>".format(task))

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    loss_function = nll_loss_on_mean
    remember_best_column = "valid_misclass"
    n_classes = 2

    if model_constraint is not None:
        assert model_constraint == 'defaultnorm'
        model_constraint = MaxNormDefaultConstraint()

    stop_criterion = MaxEpochs(max_epochs)

    set_random_seeds(seed=seed, cuda=cuda)
    if model_name == 'shallow':
        model = ShallowFBCSPNet(
            in_chans=n_chans, n_classes=n_classes,
            n_filters_time=n_start_chans,
            n_filters_spat=n_start_chans,
            input_time_length=input_time_length,
            final_conv_length=final_conv_length).create_network()
    elif model_name == 'deep':
        model = Deep4Net(
            n_chans, n_classes,
            n_filters_time=n_start_chans,
            n_filters_spat=n_start_chans,
            input_time_length=input_time_length,
            n_filters_2=int(n_start_chans * n_chan_factor),
            n_filters_3=int(n_start_chans * (n_chan_factor ** 2.0)),
            n_filters_4=int(n_start_chans * (n_chan_factor ** 3.0)),
            final_conv_length=final_conv_length,
            stride_before_pool=stride_before_pool).create_network()
    elif model_name == 'eegnet':
        model = EEGNetv4(
            n_chans, n_classes,
            input_time_length=input_time_length,
            final_conv_length=final_conv_length).create_network()
    elif model_name == "tcn":
        model = TemporalConvNet(
            input_size=n_chans,
            output_size=n_classes,
            context_size=0,
            num_channels=55,
            num_levels=5,
            kernel_size=16,
            dropout=0.05270154233150525,
            skip_mode=None,
            use_context=0,
            lasso_selection=0.0,
            rnn_normalization=None)
    else:
        assert False, "unknown model name {:s}".format(model_name)

    # maybe check if this works and wait / re-try after some time?
    # in case of all cuda devices are busy
    if cuda:
        model.cuda()

    if model_name != "tcn":
        to_dense_prediction_model(model)
    logging.info("Model:\n{:s}".format(str(model)))

    test_input = np_to_var(np.ones((2, n_chans, input_time_length, 1),
                                   dtype=np.float32))
    if list(model.parameters())[0].is_cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    if eval_folder is None:
        logging.info("will do validation")
        if lazy_loading:
            logging.info("using lazy loading to load {} recs"
                         .format(n_recordings))
            dataset = TuhLazy(train_folder, target=task,
                              n_recordings=n_recordings)
        else:
            logging.info("using traditional loading to load {} recs"
                         .format(n_recordings))
            dataset = Tuh(train_folder, n_recordings=n_recordings, target=task)

        assert not (run_on_normals and run_on_abnormals), (
            "decide whether to run on normal or abnormal subjects")
        # only run on normal subjects
        if run_on_normals:
            ids = [i for i in range(len(dataset))
                   if dataset.pathologicals[i] == 0]  # 0 is non-pathological
            dataset = TuhSubset(dataset, ids)
            logging.info("only using {} normal subjects".format(len(dataset)))
        if run_on_abnormals:
            ids = [i for i in range(len(dataset))
                   if dataset.pathologicals[i] == 1]  # 1 is pathological
            dataset = TuhSubset(dataset, ids)
            logging.info("only using {} abnormal subjects".format(len(dataset)))

        indices = np.arange(len(dataset))
        kf = KFold(n_splits=n_folds, shuffle=shuffle_folds)
        for i, (train_ind, test_ind) in enumerate(kf.split(indices)):
            assert len(np.intersect1d(train_ind, test_ind)) == 0, (
                "train and test set overlap!")

            # seed is in range of number of folds and was set by submit script
            if i == seed:
                break

        if lazy_loading:
            test_subset = TuhLazySubset(dataset, test_ind)
            train_subset = TuhLazySubset(dataset, train_ind)
        else:
            test_subset = TuhSubset(dataset, test_ind)
            train_subset = TuhSubset(dataset, train_ind)
    else:
        logging.info("will do final evaluation")
        if lazy_loading:
            train_subset = TuhLazy(train_folder, target=task)
            test_subset = TuhLazy(eval_folder, target=task)
        else:
            train_subset = Tuh(train_folder, target=task)
            test_subset = Tuh(eval_folder, target=task)

        # remove rec:
        # train/abnormal/01_tcp_ar/081/00008184/s001_2011_09_21/00008184_s001_t001
        # since it contains no crop without outliers (channels A1, A2 broken)
        subjects = [f.split("/")[-3] for f in train_subset.file_paths]
        if "00008184" in subjects:
            bad_id = subjects.index("00008184")
            train_subset = remove_file_from_dataset(
                train_subset, file_id=bad_id, file=(
                    "train/abnormal/01_tcp_ar/081/00008184/s001_2011_09_21/"
                    "00008184_s001_t001"))
        subjects = [f.split("/")[-3] for f in test_subset.file_paths]
        if "00008184" in subjects:
            bad_id = subjects.index("00008184")
            test_subset = remove_file_from_dataset(
                test_subset, file_id=bad_id, file=(
                    "train/abnormal/01_tcp_ar/081/00008184/s001_2011_09_21/"
                    "00008184_s001_t001"))

    if lazy_loading:
        iterator = LazyCropsFromTrialsIterator(
            input_time_length, n_preds_per_input, batch_size,
            seed=seed, num_workers=num_workers,
            reset_rng_after_each_batch=False,
            check_preds_smaller_trial_len=False)  # True!
    else:
        iterator = CropsFromTrialsIterator(batch_size, input_time_length,
                                           n_preds_per_input, seed)

    monitors = []
    monitors.append(LossMonitor())
    monitors.append(RAMMonitor())
    monitors.append(RuntimeMonitor())
    monitors.append(CroppedDiagnosisMonitor(input_time_length,
                                            n_preds_per_input))
    monitors.append(LazyMisclassMonitor(col_suffix='sample_misclass'))

    if lazy_loading:
        n_updates_per_epoch = len(iterator.get_batches(train_subset,
                                                       shuffle=False))
    else:
        n_updates_per_epoch = sum([1 for _ in iterator.get_batches(
            train_subset, shuffle=False)])
    n_updates_per_period = n_updates_per_epoch * max_epochs
    logging.info("there are {} updates per epoch".format(n_updates_per_epoch))

    if model_name == "tcn":
        adamw = ExtendedAdam(model.parameters(), lr=init_lr,
                             weight_decay=weight_decay, l2_decay=l2_decay,
                             gradient_clip=gradient_clip)
    else:
        adamw = AdamW(model.parameters(), init_lr,
                                  weight_decay=weight_decay)

    scheduler = CosineAnnealing(n_updates_per_period)
    optimizer = ScheduledOptimizer(scheduler, adamw, schedule_weight_decay=True)

    exp = Experiment(
        model=model,
        train_set=train_subset,
        valid_set=None,
        test_set=test_subset,
        iterator=iterator,
        loss_function=loss_function,
        optimizer=optimizer,
        model_constraint=model_constraint,
        monitors=monitors,
        stop_criterion=stop_criterion,
        remember_best_column=remember_best_column,
        run_after_early_stop=False,
        batch_modifier=None,
        cuda=cuda,
        do_early_stop=False,
        reset_after_second_run=False
    )
    return exp


def write_kwargs_and_epochs_dfs(kwargs, exp):
    result_folder = kwargs["result_folder"]
    if result_folder is None:
        return
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "config.json", "w") as json_file:
        json.dump(kwargs, json_file, indent=4, sort_keys=True)
    exp.epochs_df.to_csv(result_folder + "epochs_df_" + str(kwargs["seed"]) +
                         ".csv")


def write_predictions(y, mean_preds_per_trial, setname, kwargs, exp):
    result_folder = kwargs["result_folder"]
    if result_folder is None:
        return
    logging.info("length of y is {}".format(len(y)))
    logging.info("length of mean_preds_per_trial is {}".
                 format(len(mean_preds_per_trial)))
    assert len(y) == len(mean_preds_per_trial)

    # TODELAY: don't hardcode the mappings
    if kwargs["task"] == "pathological":
        column0, column1 = "non-pathological", "pathological"
        a_dict = {column0: mean_preds_per_trial[:, 0],
                  column1: mean_preds_per_trial[:, 1],
                  "true_pathological": y}
    # store predictions
    pd.DataFrame.from_dict(a_dict).to_csv(result_folder + "predictions_" +
                                          setname + "_" + str(kwargs["seed"]) +
                                          ".csv")


def make_final_predictions(kwargs, exp):
    exp.model.eval()
    for setname in ('train', 'test'):
        dataset = exp.datasets[setname]
        if kwargs["cuda"]:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0]).cuda()))
                               for b in exp.iterator.get_batches(dataset,
                                                                 shuffle=False)]
        else:
            preds_per_batch = [var_to_np(exp.model(np_to_var(b[0])))
                               for b in exp.iterator.get_batches(dataset,
                                                                 shuffle=False)]
        preds_per_trial = compute_preds_per_trial(
            preds_per_batch, dataset,
            input_time_length=exp.iterator.input_time_length,
            n_stride=exp.iterator.n_preds_per_input)
        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        write_predictions(dataset.y, mean_preds_per_trial, setname, kwargs, exp)


def save_model(kwargs, exp):
    result_folder = kwargs["result_folder"]
    path = result_folder + "model_{}.pt".format(kwargs["seed"])
    th.save(exp.model, path)
    path = result_folder + "state_dict_{}.pt".format(kwargs["seed"])
    th.save(exp.model.state_dict(), path)


def main():
    logging.basicConfig(level=logging.DEBUG)
    kwargs = parse_run_args()
    logging.info(kwargs)
    start_time = time.time()
    if kwargs["seed"] is None:
        assert "SLURM_ARRAY_TASK_ID" in os.environ
        kwargs["seed"] = int(os.environ["SLURM_ARRAY_TASK_ID"])
    assert kwargs["seed"] < 5, "cannot handle seed > 4, implement cv logic"
    exp = setup_exp(**kwargs)
    exp.run()
    end_time = time.time()
    run_time = end_time - start_time
    logging.info("Experiment runtime: {:.2f} sec".format(run_time))

    write_kwargs_and_epochs_dfs(kwargs, exp)
    make_final_predictions(kwargs, exp)
    save_model(kwargs, exp)


if __name__ == '__main__':
    main()
