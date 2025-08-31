# coding=utf-8
import argparse
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # main root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from tqdm import tqdm
from copy import deepcopy

import time

from utils.general import set_seed
from utils.dataloader import CustomDataLoader
from models.tsAMD_lightgts_v2 import AMD_LightGTS_v2


def main(args):
    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # workers
    torch.set_num_threads(4)
    # set seed
    set_seed(args.seed)

    # load datasets
    data_loader = CustomDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
    )

    train_data = data_loader.get_train()
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()

    # load model
    model = AMD_LightGTS_v2(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        dropout=args.dropout,
        top_k=args.top_k,
        target_patch_len=args.target_patch_len,
        d_core=args.d_core,
        alpha=args.alpha,
        target_slice=data_loader.target_slice,
        norm=args.norm,
        layernorm=args.layernorm,
        e_layers=args.e_layers
    ).to(device)

    print(sum(p.numel() for p in model.parameters()))

    # set criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-9)

    best_loss = torch.tensor(float('inf'))
    # early stopping
    early_stopping_counter = 0
    # create checkpoint directory
    save_directory = os.path.join(args.checkpoint_dir, args.name)

    if os.path.exists(save_directory):
        import glob
        import re

        path = Path(save_directory)
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        save_directory = f"{path}{n}"  # update path

    os.makedirs(save_directory)

    # start training
    for epoch in range(args.train_epochs):
        model.train()
        train_mloss = torch.zeros(1, device=device)
        iter_time = 0
        print(f"epoch : {epoch + 1}")
        print("Train")
        pbar = tqdm(enumerate(train_data), total=len(train_data))
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            start_time = time.time()
            outputs, moe_loss = model(batch_x)
            optimizer.zero_grad()
            loss = criterion(outputs, batch_y) + moe_loss
            loss.backward()
            optimizer.step()
            end_time = time.time()
            train_mloss = (train_mloss * i + loss.detach()) / (i + 1)
            pbar.set_description(('%-10s' * 1 + '%-10.8g ' * 1) % (f'{epoch + 1}/{args.train_epochs}', train_mloss))
            iteration_time = (end_time - start_time) * 1000
            iter_time = (iter_time * i + iteration_time) / (i + 1)
            # end batch -------------------------------------------------------------
        print(f"train loss: {train_mloss.item()}, iter_time: {iter_time}")

        model.eval()
        val_mloss = torch.zeros(1, device=device)
        val_mae = torch.zeros(1, device=device)
        val_mse = torch.zeros(1, device=device)
        print("Val")
        pbar = tqdm(enumerate(val_data), total=len(val_data))

        with torch.no_grad():
            for i, (batch_x, batch_y) in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, moe_loss = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_mloss = (val_mloss * i + loss.detach()) / (i + 1)
                mae = torch.abs(outputs - batch_y).mean()
                val_mae = (val_mae * i + mae.detach()) / (i + 1)
                mse = ((outputs - batch_y) ** 2).mean()
                val_mse = (val_mse * i + mse.detach()) / (i + 1)
                pbar.set_description(('%-10s' * 1 + '%-10.8g' * 1) % (f'', val_mloss))

            if val_mloss < best_loss:
                best_loss = val_mloss
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(save_directory, "best.pt"))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if epoch == args.train_epochs - 1:
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(save_directory, "best.pt"))

        print(f"val loss: {val_mloss.item()}, val MSE: {val_mse.item()}, val MAE: {val_mae.item()}")

        # early stopping check
        if early_stopping_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {args.early_stopping} epochs)")
            break

        # scheduler.step()

        # end epoch -------------------------------------------------------------

    # load best model
    model.load_state_dict(best_model)

    # start testing
    model.eval()

    test_mloss = torch.zeros(1, device=device)
    test_mae = torch.zeros(1, device=device)
    test_mse = torch.zeros(1, device=device)

    print("Final Test")
    pbar = tqdm(enumerate(test_data), total=len(test_data))

    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, moe_loss = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_mloss = (test_mloss * i + loss.detach()) / (i + 1)
            mae = torch.abs(outputs - batch_y).mean()
            test_mae = (test_mae * i + mae.detach()) / (i + 1)
            mse = ((outputs - batch_y) ** 2).mean()
            test_mse = (test_mse * i + mse.detach()) / (i + 1)
            pbar.set_description(('%-10.8g' * 1) % (test_mloss))
    print(f"test loss: {test_mloss.item()}, test MSE: {test_mse.item()}, test MAE: {test_mae.item()}")

    # 保存测试结果到txt文件
    save_test_results(args, test_mse.item(), test_mae.item())


def save_test_results(args, test_mse, test_mae):
    """
    保存测试结果到txt文件，按照指定格式
    """
    # 从数据路径中提取数据集名称
    data_path = Path(args.data)
    dataset_name = data_path.stem  # 例如 'ETTh1'

    # 创建results目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 构建结果文件名（添加LightGTS_v2标识）
    result_filename = f"{dataset_name}_lightgts_v2_results.txt"
    result_filepath = results_dir / result_filename

    # 构建模型配置信息字符串
    model_config = (
        f"AMD_LightGTS_v2_{dataset_name}_M_ft{args.seq_len}_sl{args.seq_len}_ll{args.pred_len}_"
        # f"dm{args.n_block}_nh{args.n_block}_el{args.n_block}_"
        f"dl{args.seq_len}_fc{args.feature_type}_"
        f"projection_{int(args.alpha)}_topk{args.top_k}_tpl{args.target_patch_len}_dcore{args.d_core}"
    )

    # 按照指定格式准备内容
    result_content = (
        f"{dataset_name}_{args.seq_len}_{args.pred_len}_{model_config} \n"
        f"mse:{test_mse}, mae:{test_mae}\n\n"
    )

    # 追加写入文件
    with open(result_filepath, 'a', encoding='utf-8') as f:
        f.write(result_content)

    print(f"测试结果已保存到: {result_filepath}")
    print(f"保存格式: {result_content.strip()}")


def infer_extension(dataset_name):
    if dataset_name.startswith('solar'):
        extension = 'txt'
    elif dataset_name.startswith('PEMS'):
        extension = 'npz'
    else:
        extension = 'csv'
    return extension


def parse_args():
    dataset = "ETTh1"
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    # data loader
    parser.add_argument('--data',
                        type=str,
                        default=ROOT / f'data/{dataset}.{infer_extension(dataset)}',
                        help='dataset path')
    parser.add_argument(
        '--feature_type',
        type=str,
        default='M',
        choices=['S', 'M', 'MS'],
        help=(
            'forecasting task, options:[M, S, MS]; M:multivariate predict'
            ' multivariate, S:univariate predict univariate, MS:multivariate'
            ' predict univariate'
        ),
    )
    parser.add_argument(
        '--target', type=str, default='OT', help='target feature in S or MS task'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=ROOT / 'checkpoints',
        help='location of model checkpoints',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=f'{dataset}',
        help='save best model to checkpoints/name',
    )
    # forecasting task
    parser.add_argument(
        # 96 192 336 512 672 720
        '--seq_len', type=int, default=720, help='input sequence length'
    )
    parser.add_argument(
        # 12 for PEMS  , {96, 192, 336, 720} for others
        '--pred_len', type=int, default=96, help='prediction sequence length'
    )
    # model hyperparameter

    parser.add_argument(
        '--e_layers',
        type=int,
        default=1,
        help='number of STAR encoder layers',
    )
    parser.add_argument(
        # 0.0  0.5  1.0
        '--alpha',
        type=float,
        default=0.0,
        help='compatibility parameter (not used in STAR)',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='num of periodical patch mixer top k',
    )
    parser.add_argument(
        '--target_patch_len',
        type=int,
        default=16,
        help='target patch len of periodical patch mixer',
    )
    parser.add_argument(
        '--d_core',
        type=int,
        default=None,
        help='core representation dimension for STAR module (default: seq_len)',
    )
    parser.add_argument(
        '--norm',
        type=bool,
        default=True,
        help='RevIN',
    )
    parser.add_argument(
        '--layernorm',
        type=bool,
        default=True,
        help='layernorm',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate'
    )
    # optimization
    parser.add_argument(
        '--train_epochs', type=int, default=10, help='train epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size of input data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00005,
        help='optimizer learning rate',
    )
    parser.add_argument(
        '--early_stopping', type=int, default=3, help='early stopping patience'
    )
    # save results
    parser.add_argument(
        '--result_path', default='result.csv', help='path to save result'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
