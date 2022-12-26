import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import os
from typing import List

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
from joblib import dump

# import src.models
# from src.utils import color


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=1e-4) -> None:
        self.tolearnce = tolerance
        self.min_delta = min_delta
        self.cnt = 0
        self.best_val_loss = None

    def __call__(self, val_loss):

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        elif (
            self.best_val_loss - self.min_delta
            < val_loss
            < self.best_val_loss + self.min_delta
        ):
            self.cnt += 1
        else:
            self.cnt = 0

        self.best_val_loss = min(self.best_val_loss, val_loss)

        if self.cnt >= self.tolearnce:
            return True

        return False


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c],
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size)
            )
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src = src.double()
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm

    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x - x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = (
            reconst_loss
            + self.lambda_energy * sample_energy
            + self.lambda_cov * cov_diag
        )
        return Variable(loss, requires_grad=True)

    def compute_energy(
        self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True
    ):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append(
                (
                    Cholesky.apply(cov_k.to(device) * (2 * np.pi)).diag().prod()
                ).unsqueeze(0)
            )
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2)
            * z_mu,
            dim=-1,
        )
        E_z = torch.exp(E_z)
        E_z = -torch.log(
            torch.sum(
                phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0),
                dim=1,
            )
            + eps
        )
        if sample_mean == True:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function"""
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        # phi = D
        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        # mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        (l,) = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag())
        )
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class PlanarNormalizingFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = Variable(torch.randn(1), requires_grad=True)
        self.linear = nn.Linear(dim, dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = x + self.u.to(device) * self.tanh(self.linear(x))
        return out


class Qnet(nn.Module):
    def __init__(self, in_dim=4096, hidden_dim=1024, z_dim=100, dense_dim=100):
        super(Qnet, self).__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.z_dim = z_dim
        self.e_dim = hidden_dim
        self.dense_dim = dense_dim
        self.dense = nn.Linear(self.z_dim + self.e_dim, self.dense_dim)
        self.linear_mu = nn.Linear(self.dense_dim, self.z_dim)
        self.linear_sigma = nn.Linear(self.dense_dim, self.z_dim)
        self.softplus = nn.Softplus()
        self.pnf = PlanarNormalizingFlow(dim=self.z_dim)

    def forward(self, x):
        B, W, F = x.shape

        z = torch.zeros(B, W, self.z_dim, dtype=torch.float64).to(device)
        mu, logvar = (
            torch.zeros(B, W, self.z_dim, dtype=torch.float64).to(device),
            torch.zeros(B, W, self.z_dim, dtype=torch.float64).to(device),
        )

        e_t = None
        z_t = torch.zeros(B, 1, self.z_dim).to(device)
        for t in range(W):
            # GRU
            x_t = x[:, t, :].unsqueeze(1)
            _, e_t = self.gru(x_t, e_t)

            # Dense
            dense_input = torch.cat((z_t, torch.transpose(e_t, 0, 1)), axis=2)
            dense_output = self.dense(dense_input)

            # mu and sigma sampling
            mu_t, logvar_t = self.linear_mu(dense_output), self.softplus(
                self.linear_sigma(dense_output)
            )
            mu[:, t, :], logvar[:, t, :] = mu_t.squeeze(1), logvar_t.squeeze(1)
            std = torch.exp(0.5 * logvar_t)
            z_t = mu_t + std * torch.rand_like(logvar_t)

            # latent z
            z_t = self.pnf(z_t)
            z[:, t, :] = z_t.squeeze(1)

        return z, mu, logvar


class Pnet(nn.Module):
    def __init__(self, z_dim=4096, hidden_dim=1024, dense_dim=100, out_dim=100):
        super(Pnet, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.out_dim = out_dim

        self.gru = nn.GRU(
            z_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.LGSSM = nn.Linear(z_dim, z_dim)
        self.dense = nn.Linear(self.hidden_dim, self.dense_dim)
        self.linear_mu = nn.Linear(self.dense_dim, self.out_dim)
        self.linear_sigma = nn.Linear(self.dense_dim, self.out_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        B, W, F = z.shape

        out = torch.zeros(B, W, self.out_dim, dtype=torch.float64).to(z.device)

        d_t = None

        for t in range(W):
            z_t_1 = z[:, t, :].unsqueeze(1)
            z_t = self.LGSSM(z_t_1)
            _, d_t = self.gru(z_t, d_t)

            dense_output = self.dense(torch.transpose(d_t, 0, 1))
            mu, sigma = self.linear_mu(dense_output), self.softplus(
                self.linear_sigma(dense_output)
            )
            out_t = mu + sigma * torch.rand_like(sigma)

            out[:, t, :] = out_t.squeeze(1)

        return out


class LoadTimeSeriesDataset(object):
    def __init__(
        self,
        data,
        categorical_cols: List[str],
        use_cols: List[str],
        index_col: str,
        seq_length: int,
        batch_size: int,
        train_size: float,
    ):
        """
        :param data_path: path to datafile
        :param categorical_cols: name of the categorical columns, if None pass empty list
        :param index_col: column to use as index
        :param seq_length: window length to use
        :param batch_size:
        """

        self.data = data
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(
            set(self.data.columns) - set(categorical_cols)
        )

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.train_size = train_size

        transformations = [
            (
                "scaler",
                RobustScaler(
                    with_centering=False,
                    quantile_range=(1, 99),
                ),
                self.numerical_cols,
            )
        ]
        if len(self.categorical_cols) > 0:
            transformations.append(
                ("encoder", OneHotEncoder(), self.categorical_cols)
            )
        self.preprocessor = ColumnTransformer(
            transformations, remainder="passthrough"
        )

    def preprocess_data(self):
        """Preprocessing function"""

        X_train, X_test = train_test_split(
            self.data, train_size=self.train_size, shuffle=False
        )
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        return X_train, X_test

    def frame_series(self, X, y=None):
        """
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        """

        nb_obs, nb_features = X.shape
        features = []

        for i in range(0, nb_obs - self.seq_length):
            features.append(
                torch.DoubleTensor(X[i : i + self.seq_length, :]).unsqueeze(0)
            )

        features_var = torch.cat(features)

        return TensorDataset(features_var, features_var)

    def get_loaders(self):
        """
        Preprocess and frame the dataset
        :return: DataLoaders associated to training and testing data
        """
        X_train, X_test = self.preprocess_data()
        nb_features = X_train.shape[1]

        train_dataset = self.frame_series(X_train)
        test_dataset = self.frame_series(X_test)

        train_iter = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x)
            ),
        )
        test_iter = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x)
            ),
        )
        return train_iter, test_iter, nb_features

    def invert_scale(self, predictions):
        """
        Inverts the scale of the predictions
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        unscaled = self.preprocessor.named_transformers_[
            "scaler"
        ].inverse_transform(predictions)
        return torch.Tensor(unscaled)


class LoadModel(object):
    def __init__(
        self,
        model_name,
        dim,
        data_name,
        lr,
        lrs_step_size,
        seq_len,
        retrain=False,
    ):
        self.model_name = model_name
        self.dim = dim
        self.retrain = retrain
        self.seq_len = seq_len
        self.lr = lr
        self.lrs_step_size = lrs_step_size
        self.data_name = data_name

    def get_model(self):
        model_class = getattr(src.models, self.model_name)
        model = (
            model_class(feats=self.dim, n_window=self.seq_len)
            .double()
            .to(device)
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.lrs_step_size, 0.6
        )
        fname = f"./checkpoints/{self.model_name}_{self.data_name}/model.ckpt"
        if os.path.exists(fname) and (not self.retrain):
            print(
                f"{color.GREEN}Loading pre-trained model: {self.model_name}{color.ENDC}"
            )
            checkpoint = torch.load(fname, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print(
                f"{color.GREEN}Creating new model: {self.model_name}{color.ENDC}"
            )

        return (
            model,
            optimizer,
            scheduler,
        )


def tranad_train(model, criterion, batch):

    x, target = batch

    target = target.permute(1, 0, 2)
    x = x.permute(1, 0, 2).double()
    x_hat_1, x_hat_2 = model(x.to(device))

    n = model.forward_n // 1000 + 1

    loss = criterion(x_hat_1, target) / n + (1 - 1 / n) * criterion(
        x_hat_2, target
    )

    return loss


def train(model, criterion, batch):

    model.train()
    model_name = str.lower(model.name)

    train_models = {
        "tranad": tranad_train,
    }

    assert model_name in train_models, f"Model {model.name} not implemented"

    loss = train_models[model_name](
        model=model, criterion=criterion, batch=batch
    )
    loss.backward()

    return loss


def save_model(model, optimizer, scheduler):
    folder = f"./checkpoints/{model.name}/"
    os.makedirs(folder, exist_ok=True)
    file_path = f"{folder}/model.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        file_path,
    )


def tranad_eval(model, criterion, batch):

    x, target = batch
    x = x.permute(1, 0, 2)
    _, x_hat = model(x.to(device))
    x_hat = x_hat.permute(1, 0, 2)
    loss = criterion(x_hat, target)

    return loss, x_hat, target


def eval(model, criterion, batch):
    model.eval()

    model_name = str.lower(model.name)

    eval_models = {
        "tranad": tranad_eval,
    }

    assert model_name in eval_models, f"Model {model.name} not implemented"

    loss, x_hat, target = eval_models[model_name](
        model=model,
        criterion=criterion,
        batch=batch,
    )

    return loss, x_hat, target