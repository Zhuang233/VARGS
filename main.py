import os
import argparse
from utils.config import get_config
from tools import builder
from torchsummary import summary
from utils.gaussian import write_gaussian_feature_to_ply, unnormalize_gaussians
from datasets import data_transforms
from utils.AverageMeter import AverageMeter
import time
import numpy as np



class Acc_Metric:
    def __init__(self, acc=0.0):
        if type(acc).__name__ == "dict":
            self.acc = acc["acc"]
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict["acc"] = self.acc
        return _dict

parser = argparse.ArgumentParser() 
args = parser.parse_args()


args.resume = False
args.local_rank = 0
args.config = "./cfgs/aaa.yaml"
args.experiment_path = "./experiment/"
args.distributed = False
args.num_workers = 4
args.start_ckpts = None

config = get_config(args)
config.dataset.test.others.bs = 1
config.dataset.train.others.bs = 1

train_transforms = data_transforms.PointcloudScaleAndTranslate()

if __name__ == '__main__':
    ## 加载数据集
    _, dataloader_test = builder.dataset_builder(args, config.dataset.test)
    _, dataloader_train = builder.dataset_builder(args, config.dataset.train)

    # for taxonomy_id, model_id,  data, scale_c, scale_m in dataloader_test:
    #     print(model_id)

    ## 加载模型
    model = builder.model_builder(config.model)
    # summary(model, input_size=(1,14336))

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.0)
    metrics = Acc_Metric(0.0)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(
            model, args, strict_load=True
        )
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(
            model, args.start_ckpts, strict_load=True
        )

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args)


    model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(["Loss"])

        n_batches = len(dataloader_train)
        num_iter = 0

        model.train()
        
        model.zero_grad()
        for idx, (taxonomy_ids, model_ids, data, scale_c, scale_m) in enumerate(dataloader_train):
            num_iter += 1
            data_time.update(time.time() - batch_start_time)

            points = data


            # if (epoch == config.max_epoch and idx % 50 == 0):  # save last epoch ply for visualization
            if (epoch == config.max_epoch):  # save last epoch ply for visualization
                loss_dict, vis_gaussians, full_rebuild_gaussian, original_gaussians = (model(points, save=True))

                # save to gaussian ply
                os.makedirs(os.path.join(args.experiment_path, "save_ply"), exist_ok=True)

                original_gaussians, vis_gaussians, full_rebuild_gaussian = (
                    unnormalize_gaussians(
                        original_gaussians,
                        vis_gaussians,
                        full_rebuild_gaussian,
                        scale_c,
                        scale_m,
                        config,
                    )
                )

                for i in range(vis_gaussians.shape[0]):  # save whole batch
                    vis_gaussians_ply_path = os.path.join(
                        args.experiment_path,
                        "save_ply",
                        f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_vis_gaussians.ply",
                    )
                    full_rebuild_gaussian_ply_path = os.path.join(
                        args.experiment_path,
                        "save_ply",
                        f"{model_ids[i]}_ep_{str(epoch).zfill(4)}_full_rebuild_gaussian.ply",
                    )
                    original_gaussians_ply_path = os.path.join(
                        args.experiment_path,
                        "save_ply",
                        f"{model_ids[i]}_original_gaussians.ply",
                    )
                    write_gaussian_feature_to_ply(
                        vis_gaussians[i], vis_gaussians_ply_path
                    )
                    write_gaussian_feature_to_ply(
                        full_rebuild_gaussian[i], full_rebuild_gaussian_ply_path
                    )
                    write_gaussian_feature_to_ply(
                        original_gaussians[i], original_gaussians_ply_path
                    )
            else:
                if epoch != config.max_epoch:
                    points = train_transforms.augument(
                        points, attribute=config.model.attribute
                    )
                loss_dict = model(points)

                        # aggregate all loss
            
            loss = sum([loss_dict[key] for key in loss_dict.keys()])
            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                model.zero_grad()

            losses.update([loss.mean().item()])

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:

                print(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f"
                    % (
                        epoch,
                        config.max_epoch,
                        idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        ["%.4f" % l for l in losses.val()],
                        optimizer.param_groups[0]["lr"],
                    )
                )
                # print loss_dict
                for key in loss_dict.keys():
                    print(f"{key} = {loss_dict[key]}")

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        builder.save_checkpoint(
            model,
            optimizer,
            epoch,
            metrics,
            best_metrics,
            "ckpt-last",
            args
        )










