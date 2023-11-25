import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
import copy
from .Functions import *
from .evaluation import *

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, **kwargs):
        return torch.add(self.module(x, **kwargs), x)

class Bindingscore_head(nn.Module):
    def __init__(self,
                 n_filters,
                 kernel_size=[75],
                 pool_size=10,
                 upsample_size=10,
                 pool_mode='avg'
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = [k // 2 for k in kernel_size]
        self.pool_size = pool_size
        self.upsample_size=upsample_size
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=n_filters,
                        out_channels=1,
                        kernel_size=k,
                        padding=p,
                        )
            for k, p in zip(kernel_size, self.padding)
        ])
        if pool_mode == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        elif pool_mode == 'max':
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.upsampling = nn.Upsample(scale_factor=upsample_size, mode='nearest')

    def forward(self, X, output_len=None, upsample=True):
        X_bindingscore = torch.cat([conv(X) for conv in self.conv_layers], dim=1)
        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len / self.upsample_size * self.pool_size)
            trim = (X_bindingscore.shape[-1] - output_len_needed_in_X) // 2
        # print (output_len, X_bindingscore.shape, output_len_needed_in_X, trim)
        if trim > 0:
            X_bindingscore = X_bindingscore[:, :, trim:-trim]

        X_bindingscore = self.pool(X_bindingscore)
        if upsample:
            X_bindingscore = self.upsampling(X_bindingscore)

        return X_bindingscore

class Bindingscore_headv2(nn.Module):
    def __init__(self,
                 n_filters,
                 kernel_size=[75],
                 pool_size=10,
                 upsample_size=10,
                 pool_mode='avg'
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = [k // 2 for k in kernel_size]
        self.pool_size = pool_size
        self.upsample_size=upsample_size
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=n_filters,
                        out_channels=1,
                        kernel_size=k,
                        padding=p,
                        )
            for k, p in zip(kernel_size, self.padding)
        ])
        if pool_mode == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        elif pool_mode == 'max':
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.upsampling = nn.Upsample(scale_factor=upsample_size, mode='nearest')
        self.linear = nn.Linear(n_filters, len(kernel_size), bias=True)
    def forward(self, X, output_len=None, upsample=True):
        X_count = self.linear(X.mean(dim=-1))
        X_bindingscore = torch.cat([conv(X) for conv in self.conv_layers], dim=1)
        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len / self.upsample_size * self.pool_size)
            trim = (X_bindingscore.shape[-1] - output_len_needed_in_X) // 2
        # print (output_len, X_bindingscore.shape, output_len_needed_in_X, trim)
        if trim > 0:
            X_bindingscore = X_bindingscore[:, :, trim:-trim]

        X_bindingscore = self.pool(X_bindingscore)


        if upsample:
            X_bindingscore = self.upsampling(X_bindingscore)

        return X_bindingscore, X_count.unsqueeze(-1)

@torch.no_grad()
def validation_step_footprint(model,
                    validation_data,
                    validation_size,
                    mode=['classification', 'classification', 'regression'],
                    verbose=False):
    device = next(model.parameters()).device
    size = 0
    profile_pearson = []
    across_batch_pearson = [[], []]
    val_loss = [0] * len(mode)
    for data in tqdm(validation_data, disable=not verbose):
        X, y = data
        pred_score = model.predict(X, batch_size=X.shape[0]).to(device)
        y = y.to(device)
        for i in range(y.shape[1]):
            if mode[i] == 'classification':
                loss_ = F.binary_cross_entropy_with_logits(pred_score[:, i].view(-1), y[:, i].view(-1))
                pred_score[:, i] = torch.sigmoid(pred_score[:, i])
            elif mode[i] == 'regression':
                loss_ = F.mse_loss(pred_score[:, i], y[:, i])
            elif mode[i] == 'shape':
                loss_ = shape_loss(pred_score[:, i], y[:, i])
                pred_score[:, i] = torch.softmax(pred_score[:, i], dim=-1)
            else:
                print (mode, y.shape)
            val_loss[i] += loss_.item()
        corr = batch_pearson_correlation(pred_score,
                                                    y).detach().cpu()
        across_batch_pearson[0].append(pred_score.detach().cpu())
        across_batch_pearson[1].append(y.detach().cpu())
        profile_pearson.append(corr)

        size += 1
        if validation_size is not None and size > validation_size:
            break

    val_loss = [l / size for l in val_loss]
    profile_pearson = torch.cat(profile_pearson, dim=0).mean(dim=0).detach().cpu().numpy()
    pred_score, y = torch.cat(across_batch_pearson[0], dim=0), torch.cat(across_batch_pearson[1], dim=0)
    pred_score = torch.permute(pred_score, (0, 2, 1)).reshape((-1, pred_score.shape[1]))
    y = torch.permute(y, (0, 2, 1)).reshape((-1, y.shape[1]))
    # print (pred_score.shape, y.shape)
    return val_loss, profile_pearson, batch_pearson_correlation(pred_score.T,
                                                                y.T).detach().cpu()


class BindingScoreBPNet(nn.Module):
    """
    This defines the bindingnet model
    """

    def __init__(self,
                 dna_cnn_model=None,
                 hidden_layer_model=None,
                 profile_cnn_model=None,
                 output_len=1000,
                 res=1):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.output_len = output_len
        self.res=res
        self.upsample=True


    def forward(self, X, output_len=None):
        if output_len is None:
            output_len = self.output_len
        res = self.res
        # get the motifs!
        # print (X.shape, output_len)
        X = self.dna_cnn_model(X)

        # get the hidden layer
        X = self.hidden_layer_model(X)
        # print (X.shape)
        # trim = (X.shape[-1] - output_len // res)  // 2
        # print(trim, X.shape)
        # get the profile
        score = self.profile_cnn_model(X,
                                       output_len=output_len,
                                       upsample=self.upsample)
        # if score.shape[-1] != output_len // res:
        #     score = score[:, :, :-1]
        # print (score.shape, trim)
        if res > 1:
            score = torch.repeat_interleave(score, res, dim=-1)
        # print(score.shape)
        return score

    def fit(self, training_data,
            validation_data=None,
            validation_size=None,
            mode=['classification', 'classification', 'regression'],
            max_epochs=100,
            optimizer=None,
            scheduler=None,
            validation_freq=100,
            early_stopping=False,
            return_best=True, savename='model'):
        """
        This is the fit function for BPNet
        Parameters
        ----------
        training_data
        validation_data
        max_epochs
        batch_size
        optimizer
        scheduler
        validation_freq
        early_stopping

        Returns
        -------

        """

        early_stopping_counter = 0
        best_loss = np.inf
        device = next(self.parameters()).device

        assert validation_size is not None or validation_data is not None, "Either validation_size or validation_data should be provided"

        if validation_data is None:
            validation_data = training_data
        if validation_size is None:
            validation_size = len(validation_data)
        if validation_freq is None:
            validation_freq = len(training_data)
        for epoch in range(max_epochs):
            bar = trange(validation_freq, desc=' - (Training) {epoch}'.format(
                epoch=epoch + 1), leave=False)
            moving_avg_loss = 0
            iteration = 0
            training_data_epoch_loader = training_data.resample()
            for data in training_data_epoch_loader:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pred_score = self.forward(X)
                loss = 0
                desc_str = ' - (Training) {epoch}'.format(
                    epoch=epoch + 1)
                for i in range(y.shape[1]):
                    if mode[i] == 'classification':
                        loss_ = F.binary_cross_entropy_with_logits(pred_score[:, i].view(-1), y[:, i].view(-1))
                    elif mode[i] == 'regression':
                        loss_ = F.mse_loss(pred_score[:, i], y[:, i])
                    elif mode[i] == 'shape':
                        loss_ = shape_loss(pred_score[:, i], y[:, i])
                    loss += loss_
                    desc_str+= " Loss_{i}: {loss:.2f}".format(i=i, loss=loss_.item())
                loss.backward()
                optimizer.step()

                bar.set_description(desc_str)
                bar.update(1)
                iteration += 1
                moving_avg_loss += loss.item()
                if iteration >= validation_freq:
                    break

            print(' - (Training) {epoch} Loss: {loss:.2f}'.format(
                    epoch=epoch + 1,
                    loss=moving_avg_loss / iteration))

            bar.close()
            self.eval()
            val_loss, profile_pearson, across_pearson = validation_step_footprint(self,
                                                       validation_data,
                                                       validation_size,
                                                       mode=mode,
                                                        verbose=True)
            self.train()
            val_loss_all = np.sum(val_loss)
            print(' - (Validation) {epoch} \
            Loss: {loss:.2f}'.format(
                epoch=epoch + 1, loss=val_loss_all))
            print ("Profile pearson", profile_pearson)
            print("Across peak pearson", across_pearson)

            if val_loss_all < best_loss:
                print ("best loss", val_loss_all)
                best_loss = val_loss_all
                early_stopping_counter = 0
                torch.save(self.state_dict(), savename)
                torch.save(self, savename + '.model.pt')
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    print('Early stopping')
                    break

        if return_best:
            self.load_state_dict(torch.load(savename))
            print ("loaded best model")




        return epoch



    def predict(self, X, batch_size=64, verbose=False):
        """
        This is the predict function for BPNet
        """
        pred_scores = []
        states = []
        for name, module in self.named_modules():
            if verbose:
                print(name, "training", module.training)
            states.append(module.training)
            module.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            for i in trange(0, len(X), batch_size, disable=not verbose):
                X_batch = X[i:i + batch_size].to(device)
                X_score = self.forward(X_batch)
                pred_scores.append(X_score.detach().cpu())

            pred_scores = torch.cat(pred_scores, dim=0)

        for (name, module), state in zip(self.named_modules(), states):
            if state:
                if verbose:
                    print("flip", name, "back to training")
                module.train()

        return pred_scores



class BindingScoreBPNetv2(nn.Module):
    """
    This defines the bindingnet model
    """

    def __init__(self,
                 dna_cnn_model=None,
                 hidden_layer_model=None,
                 profile_cnn_model=None,
                 output_len=1000,
                 res=1):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.output_len = output_len
        self.res=res
        self.upsample=True


    def forward(self, X, output_len=None):
        if output_len is None:
            output_len = self.output_len
        res = self.res
        # get the motifs!
        # print (X.shape, output_len)
        X = self.dna_cnn_model(X)

        # get the hidden layer
        X = self.hidden_layer_model(X)
        # print (X.shape)
        # trim = (X.shape[-1] - output_len // res)  // 2
        # print(trim, X.shape)
        # get the profile
        shape, score = self.profile_cnn_model(X,
                                       output_len=output_len,
                                       upsample=self.upsample)
        # if score.shape[-1] != output_len // res:
        #     score = score[:, :, :-1]
        # print (score.shape, trim)
        if res > 1:
            shape = torch.repeat_interleave(shape, res, dim=-1)
        # print(score.shape)
        return shape, score

    def fit(self, training_data,
            validation_data=None,
            validation_size=None,
            mode=['classification', 'classification', 'regression'],
            max_epochs=100,
            optimizer=None,
            scheduler=None,
            validation_freq=100,
            early_stopping=False,
            return_best=True, savename='model'):
        """
        This is the fit function for BPNet
        Parameters
        ----------
        training_data
        validation_data
        max_epochs
        batch_size
        optimizer
        scheduler
        validation_freq
        early_stopping

        Returns
        -------

        """

        early_stopping_counter = 0
        best_loss = np.inf
        device = next(self.parameters()).device

        assert validation_size is not None or validation_data is not None, "Either validation_size or validation_data should be provided"

        if validation_data is None:
            validation_data = training_data
        if validation_size is None:
            validation_size = len(validation_data)
        if validation_freq is None:
            validation_freq = len(training_data)
        for epoch in range(max_epochs):
            bar = trange(validation_freq, desc=' - (Training) {epoch}'.format(
                epoch=epoch + 1), leave=False)
            moving_avg_loss = 0
            iteration = 0
            training_data_epoch_loader = training_data.resample()
            for data in training_data_epoch_loader:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pred_shape, pred_score = self.forward(X)

                pred_score = F.softmax(pred_shape, dim=-1) * y.shape[-1] * pred_score
                loss = 0
                desc_str = ' - (Training) {epoch}'.format(
                    epoch=epoch + 1)
                for i in range(y.shape[1]):
                    if mode[i] == 'classification':
                        loss_ = F.binary_cross_entropy_with_logits(pred_score[:, i].view(-1), y[:, i].view(-1))
                    elif mode[i] == 'regression':
                        loss_ = F.mse_loss(pred_score[:, i], y[:, i])
                    elif mode[i] == 'shape':
                        loss_ = shape_loss(pred_score[:, i], y[:, i])
                    loss += loss_
                    desc_str+= " Loss_{i}: {loss:.2f}".format(i=i, loss=loss_.item())
                loss.backward()
                optimizer.step()

                bar.set_description(desc_str)
                bar.update(1)
                iteration += 1
                moving_avg_loss += loss.item()
                if iteration >= validation_freq:
                    break

            print(' - (Training) {epoch} Loss: {loss:.2f}'.format(
                    epoch=epoch + 1,
                    loss=moving_avg_loss / iteration))

            bar.close()
            self.eval()
            val_loss, profile_pearson, across_pearson = validation_step_footprint(self,
                                                       validation_data,
                                                       validation_size,
                                                       mode=mode,
                                                        verbose=True)
            self.train()
            val_loss_all = np.sum(val_loss)
            print(' - (Validation) {epoch} \
            Loss: {loss:.2f}'.format(
                epoch=epoch + 1, loss=val_loss_all))
            print ("Profile pearson", profile_pearson)
            print("Across peak pearson", across_pearson)

            if val_loss_all < best_loss:
                print ("best loss", val_loss_all)
                best_loss = val_loss_all
                early_stopping_counter = 0
                torch.save(self.state_dict(), savename)
                torch.save(self, savename + '.model.pt')
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    print('Early stopping')
                    break

        if return_best:
            self.load_state_dict(torch.load(savename))
            print ("loaded best model")




        return epoch



    def predict(self, X, batch_size=64, verbose=False):
        """
        This is the predict function for BPNet
        """
        pred_scores = []
        states = []
        for name, module in self.named_modules():
            if verbose:
                print(name, "training", module.training)
            states.append(module.training)
            module.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            for i in trange(0, len(X), batch_size, disable=not verbose):
                X_batch = X[i:i + batch_size].to(device)
                X_shape, X_score = self.forward(X_batch)
                X_score = F.softmax(X_shape, dim=-1) * X_shape.shape[-1] * X_score
                pred_scores.append(X_score.detach().cpu())

            pred_scores = torch.cat(pred_scores, dim=0)

        for (name, module), state in zip(self.named_modules(), states):
            if state:
                if verbose:
                    print("flip", name, "back to training")
                module.train()

        return pred_scores