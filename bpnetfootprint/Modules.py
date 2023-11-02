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
        return self.module(x, **kwargs) + x

class RelMultiHeadAttention(nn.Module):
    ''' Relative Multi-Head Attention module '''

    def __init__(
            self,
            input_dim,
            n_rel_pos_features,
            n_head,
            d_model,
            d_k,
            d_v,
            dropout,
            pos_dropout):
        super().__init__()

        self.scale = d_k ** -0.5
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.w_out = nn.Linear(n_head * d_v, d_model)
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

        self.n_rel_pos_features = n_rel_pos_features
        # The nn that takes rel_pos_features -> multi-head rel_pos_keys, that'll attention with querys
        self.w_rel_pos_k = nn.Linear(self.n_rel_pos_features, self.d_k * self.n_head, bias=False)
        # two bias term injected. similar to transformer-XL
        self.rel_embed_bias = nn.Parameter(torch.randn(1, self.n_head, 1, self.d_k))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, self.n_head, 1, self.d_k))

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        if k is None:
            k = q
        if v is None:
            v = q

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(0, 2, 1, 3) * self.scale  # sz_b, n_head, len_q, dq
        k = k.permute(0, 2, 1, 3)  # sz_b, n_head, len_k, dk
        v = v.permute(0, 2, 1, 3)  # sz_b, n_head, len_v, dv

        attn = torch.einsum('bnqd,bnkd->bnqk', q + self.rel_embed_bias, k)  # sz_b, n_head, len_q, len_k

        # relative position encoding
        positions = self.pos_dropout(get_positional_embed(len_k, self.n_rel_pos_features, k.device))
        rel_pos_k = self.w_rel_pos_k(positions).view(-1, n_head, d_k).permute(1, 0, 2)  # n_head, len_k, d_k
        attn_pos = torch.einsum('bnqd,nkd->bnqk', q + self.rel_pos_bias, rel_pos_k)
        # shift to get the relative attention
        attn_pos = relative_shift(attn_pos)

        attn = torch.softmax(attn + attn_pos, dim=-1)
        attn = self.attn_dropout(attn)  # sz_b, n_head, len_q, len_k
        out = torch.einsum('bnqk, bnkd->bnqd', attn, v)  # sz_b, n_head, len_q, dv
        out = self.w_out(out.permute(0, 2, 1, 3).contiguous().view(sz_b, len_q, -1))  # sz_b, len_q, d_model
        return out

class Pass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class DilatedCNN(nn.Module):
    """
    This part only takes into account the Dilated CNN stack
    """

    def __init__(self,
               n_filters=64,
               n_layers=6,
               kernel_size=3,
               activation=nn.ReLU(),
               batch_norm=False,
               residual=True
               ):
        super().__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.activation = activation
        self.batch_norm = batch_norm
        self.residual = residual

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=n_filters,
                      out_channels=n_filters,
                      kernel_size=kernel_size,
                      dilation=2 ** (i+1),
                      padding=2 ** (i+1),)
            for i in range(n_layers)
        ])

        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(n_filters) if batch_norm else Pass()
            for i in range(n_layers)
        ])

        self.activation_layers = nn.ModuleList([
            copy.deepcopy(activation)
            for i in range(n_layers)
        ])

        return

    def forward(self, X):
        '''
        Parameters
        ----------
        self
        X: torch.tensor, shape=(batch_size, n_filters, seq_len)

        Returns
        -------

        '''
        for i in range(self.n_layers):
            X_conv = self.conv_layers[i](X)
            X_conv = self.batch_norm_layers[i](X_conv)
            X_conv = self.activation_layers[i](X_conv)
            if self.residual:
                X = X + X_conv
        return X

class DNA_CNN(nn.Module):
    """
    This is actually as simple as one CNN layer,
    It's used to extract the DNA sequence features (the first layer)
    just to keep the consistency using the Module way of construction
    """
    def __init__(self,
                 n_filters=64,
                 kernel_size=21,
                 padding=10,
                 activation=nn.ReLU(),
                 ):

        super().__init__()
        
        self.conv = nn.Conv1d(in_channels=4,
                                out_channels=n_filters,
                                kernel_size=kernel_size,
                                padding=padding)
        self.activation = activation

    def forward(self, X):
        X = self.conv(X)
        X = self.activation(X)
        return X

class BPNet_head(nn.Module):
    def __init__(self,
                 n_output,
                 n_filters,
                 kernel_size=75,
                 profile_output_bias=True,
                 count_output_bias=True,):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels=n_filters,
                                out_channels=n_output,
                                kernel_size=kernel_size,
                                padding=self.padding,
                                bias=profile_output_bias)

        self.linear = nn.Linear(in_features=n_filters,
                                out_features=n_output,
                                bias=count_output_bias)


    def forward(self, X, trim=0):
        # print (trim, -trim,trim-self.padding, -trim+self.padding)
        X_profile = self.conv(X)[:, :, trim:-trim]
        # [:, :, trim-self.padding:-trim+self.padding]
        X_count = self.linear(X.mean(dim=-1))
        return X_profile, X_count

class Footprint_head(nn.Module):
    def __init__(self,
                 n_scales,
                 n_filters,
                 kernel_size=75,
                 use_bias=True,):

        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels=n_filters,
                                out_channels=n_scales,
                                kernel_size=kernel_size,
                                padding=self.padding,
                                bias=use_bias)

    def forward(self, X, trim=0):
        X_footprint = self.conv(X)[:, :, trim:-trim]
        return X_footprint

class Bindingscore_head(nn.Module):
    def __init__(self,
                 n_filters,
                 kernel_size=[75],
                 pool_size=10,
                 pool_mode='avg'
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = [k // 2 for k in kernel_size]

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

    def forward(self, X, trim=0):
        X_bindingscore = torch.cat([conv(X)[:, :, trim:-trim] for conv in self.conv_layers], dim=1)
        X_bindingscore = self.pool(X_bindingscore)
        return X_bindingscore

@torch.no_grad()
def validation_step_footprint(model,
                    validation_data,
                    validation_size,
                    mode=['classification', 'classification', 'regression']):
    size = 0
    profile_pearson = []
    val_loss = [0] * len(mode)
    for data in validation_data:
        X, y = data
        pred_score = model.predict(X, batch_size=X.shape[0])

        for i in range(y.shape[1]):
            if mode[i] == 'classification':
                loss_ = F.binary_cross_entropy_with_logits(pred_score[:, i], y[:, i])
            elif mode[i] == 'regression':
                loss_ = F.mse_loss(pred_score[:, i], y[:, i])
            val_loss[i] += loss_.item()
        corr = batch_pearson_correlation(pred_score,
                                                    y)
        profile_pearson.append(corr)


        size += 1
        if validation_size is not None and size >= validation_size:
            break

    val_loss = [l / size for l in val_loss]
    profile_pearson = torch.cat(profile_pearson, dim=0).mean(dim=0).detach().cpu().numpy()

    return val_loss, profile_pearson


class BindingScoreBPNet(nn.Module):
    """
    This defines the bindingnet model
    """

    def __init__(self,
                 dna_cnn_model=None,
                 hidden_layer_model=None,
                 profile_cnn_model=None,
                 input_len=2114,
                 output_len=1000):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model

        self.input_len = input_len
        self.output_len = output_len


    def forward(self, X):
        # get the motifs!
        X = self.dna_cnn_model(X)

        # get the hidden layer
        X = self.hidden_layer_model(X)

        trim = (X.shape[-1] - self.output_len) // 2
        # get the profile
        score = self.profile_cnn_model(X, trim=trim)

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
                        loss_ = F.binary_cross_entropy_with_logits(pred_score[:, i], y[:, i])
                    elif mode[i] == 'regression':
                        loss_ = F.mse_loss(pred_score[:, i], y[:, i])

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

            val_loss, profile_pearson = validation_step_footprint(self,
                                                       validation_data,
                                                       validation_size,
                                                       mode=mode)
            val_loss_all = np.sum(val_loss)
            print(' - (Validation) {epoch} \
            Loss: {loss:.2f}'.format(
                epoch=epoch + 1, loss=val_loss_all))
            print ("Profile pearson", profile_pearson)

            if val_loss_all < best_loss:
                print ("best loss", val_loss_all)
                best_loss = val_loss_all
                early_stopping_counter = 0
                torch.save(self.state_dict(), savename)
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    print('Early stopping')
                    break

        if return_best:
            self.load_state_dict(torch.load(savename))
            print ("loaded best model")




        return



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


@torch.no_grad()
def validation_step(model,
                    validation_data,
                    validation_size):
    size = 0
    y_count_all = []
    X_count_all = []
    profile_pearson = []
    count_mse = 0
    val_profile_loss = 0
    for data in validation_data:
        X, y = data
        pred_profile, pred_count = model.predict(X, batch_size=X.shape[0])
        X_count_all.append(pred_count)
        pred_profile = F.log_softmax(pred_profile, dim=-1)
        y_count = y.sum(dim=-1)
        y_count_all.append(y_count)

        val_profile_loss += multinomial_negative_log_likelihood(pred_profile.reshape(pred_profile.shape[0], -1),
                                                       y.reshape((y.shape[0], -1))).sum().item()
        profile_pearson.append(batch_pearson_correlation(pred_profile,
                                                    y))
        mse_loss = log1p_mse(pred_count, y_count, reduction='sum').sum().item()
        count_mse += mse_loss


        size += 1
        if validation_size is not None and size >= validation_size:
            break

    X_count_all = torch.cat(X_count_all, dim=0)
    y_count_all = torch.cat(y_count_all, dim=0)
    profile_pearson = torch.cat(profile_pearson, dim=0).mean().item()
    val_profile_loss = val_profile_loss / len(X_count_all)
    count_mse = count_mse / len(X_count_all)
    count_pearson = batch_pearson_correlation(X_count_all.T,
    torch.log1p(y_count_all).T).mean().item()

    return val_profile_loss, count_mse, profile_pearson, count_pearson,

class BPNetBase(nn.Module):
    """
    This defines the Base neural network for BPNet
    It defines nothing about the structure, forward flow, more or less about the fit and predict
    It is separated such that BPNet class and ChromBPNet class can both inherit from this class
    while having different network structure
    """

    def __init__(self):
        super().__init__()
        return

    def forward(self, X):
        raise NotImplementedError

    def fit(self, training_data,
            validation_data=None,
            validation_size=None,
            count_loss_weight=10.0,
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
            for data in training_data:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pred_profile, pred_count = self.forward(X)
                pred_profile = F.log_softmax(pred_profile, dim=-1)
                pred_profile = pred_profile.reshape(pred_profile.shape[0], -1)

                profile_loss = multinomial_negative_log_likelihood(pred_profile, y.reshape((y.shape[0], -1))).mean()
                count_loss = log1p_mse(pred_count, y.sum(dim=-1)).mean()

                loss = (profile_loss + count_loss_weight * count_loss)
                loss.backward()
                optimizer.step()

                bar.set_description(
                    ' - (Training) {epoch} Loss: {loss:.2f} Loss_profile: {profile_loss:.2f} Loss_count: {count_loss:.2f}'.format(
                        epoch=epoch + 1,
                        loss=loss.item(),
                        profile_loss=profile_loss.item(),
                        count_loss=count_loss.item()))
                bar.update(1)
                iteration += 1
                moving_avg_loss += np.array([loss.item(),
                                             profile_loss.item(),
                                             count_loss.item()])
                if iteration >= validation_freq:
                    break
            print(
                ' - (Training) {epoch} Loss: {loss:.2f} Loss_profile: {profile_loss:.2f} Loss_count: {count_loss:.2f}'.format(
                    epoch=epoch + 1,
                    loss=moving_avg_loss[0] / iteration,
                    profile_loss=moving_avg_loss[1] / iteration,
                    count_loss=moving_avg_loss[2] / iteration))

            bar.close()

            val_profile_loss, count_mse, profile_pearson, count_pearson = validation_step(self,
                                                                                          validation_data,
                                                                                          validation_size)
            val_loss = val_profile_loss + count_loss_weight * count_mse

            print(' - (Validation) {epoch} \
            Loss: {loss:.2f} \
            Profile Loss: {profile_loss:.2f} \
            Count mse: {count_mse:.2f} \
            Profile pearson: {profile_pearson:.2f} \
            Count pearson: {count_pearson:.2f} \
            '.format(
                epoch=epoch + 1, loss=val_loss,
                profile_loss=val_profile_loss,
                profile_pearson=profile_pearson,
                count_pearson=count_pearson,
                count_mse=count_mse))
            if val_loss < best_loss:
                print("best loss", val_loss)
                best_loss = val_loss
                early_stopping_counter = 0
                torch.save(self.state_dict(), savename)
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    print('Early stopping')
                    break

        if return_best:
            self.load_state_dict(torch.load(savename))
            print("loaded best model")

        return

    def predict(self, X, batch_size=64, verbose=False):
        """
        This is the predict function for BPNet
        """
        pred_profiles, pred_counts = [], []
        states = []
        for name, module in self.named_modules():
            if verbose:
                print (name, "training", module.training)
            states.append(module.training)
            module.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            for i in trange(0, len(X), batch_size, disable=not verbose):
                X_batch = X[i:i+batch_size].to(device)
                X_profile, X_count = self.forward(X_batch)
                pred_profiles.append(X_profile.detach().cpu())
                pred_counts.append(X_count.detach().cpu())

            pred_profiles = torch.cat(pred_profiles, dim=0)
            pred_counts = torch.cat(pred_counts, dim=0)

        for (name, module), state in zip(self.named_modules(), states):
            if state:
                if verbose:
                    print ("flip", name, "back to training")
                module.train()

        return pred_profiles, pred_counts

class BPNet(BPNetBase):
    """
    This defines the BPNet way of doing count prediction & profile prediction
    But it still gives you the flexibility to change the model architecture, even utilizing transformers
    """

    def __init__(self,
                 dna_cnn_model=None,
                 hidden_layer_model=None,
                 profile_cnn_model=None,
                 input_len=2114,
                 output_len=1000):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model

        self.input_len = input_len
        self.output_len = output_len


    def forward(self, X):
        # get the motifs!
        X = self.dna_cnn_model(X)

        # get the hidden layer
        X = self.hidden_layer_model(X)

        trim = (X.shape[-1] - self.output_len) // 2
        # get the profile
        X_profile, X_count = self.profile_cnn_model(X, trim=trim)

        return X_profile, X_count

class ChromBPNet(BPNetBase):
    """
    This defines the ChromBPNet model with a bias model & an accessibility model,
    and how to integrate them into one prediction
    Again, this gives you the flexibility to change the architecture

    I let ChromBPNet inherit from BPNet,
    because I want to reuse the fit function
    """

    def __init__(self,
                 bias_model=None,
                 accessibility_model=None,
                 fix_bias_model=True):
        super().__init__()
        if fix_bias_model:
            bias_model.eval()
            for param in bias_model.parameters():
                param.requires_grad = False

        self.bias_model = bias_model
        self.accessibility_model = accessibility_model


    def forward(self,X):
        profile_bias, count_bias  = self.bias_model(X)
        profile_acc, count_acc = self.accessibility_model(X)
        profile = profile_acc + profile_bias
        counts = torch.logsumexp(torch.stack([count_acc, count_bias]),
                                   dim=0)
        return profile, counts
