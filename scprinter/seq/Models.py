from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm, trange

from .Baseline_Modules import *
from .Functions import *
from .minimum_footprint import *
from .Modules import *


def predict(self, X, batch_size=64, verbose=False):
    """
    This is the predict function
    """
    pred_footprints = []
    pred_scores = []
    self.eval()
    device = next(self.parameters()).device
    with torch.no_grad():
        for i in trange(0, len(X), batch_size, disable=not verbose):
            X_batch = X[i : i + batch_size].to(device)
            X_foot, X_score = self(X_batch)
            pred_footprints.append(X_foot.detach().cpu())
            pred_scores.append(X_score.detach().cpu())
        pred_footprints = torch.cat(pred_footprints, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)
    return pred_footprints, pred_scores


@torch.no_grad()
def validation_step_footprint(
    model, validation_data, validation_size, dispmodel, modes, linear_correction=False
):
    device = next(model.parameters()).device
    size = 0
    profile_pearson = []
    across_batch_pearson = [[], []]
    across_batch_pearson_coverage = [[], []]
    val_loss = [0]

    mean_pred_score, mean_y, mean_pred_coverage, mean_coverage = 0, 0, 0, 0
    if validation_size is None:
        validation_size = len(validation_data)
    bar = trange(validation_size, desc=" - (Validation)", leave=False, dynamic_ncols=True)
    total_len = 0
    for data in validation_data:
        if len(data) == 2:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            cell = None
            norm_cov = None
        else:
            X, y, cell, _, norm_cov = data
            cell = cell[..., 0]
            X = X.to(device)
            y = y.to(device)
            cell = cell.to(device)
            # norm_cov = norm_cov.to(device)
            norm_cov = None

        pred_score, pred_coverage = model(X, cell)
        coverage = torch.log1p(y[:, 0].sum(dim=-1)) if norm_cov is None else norm_cov
        # print (coverage.min(), coverage.max())
        y = multiscaleFoot(y[:, 0], y[:, 1], modes, dispmodel)
        mask = ~torch.isnan(y)
        # y = y - torch.nansum(y, dim=(1, 2), keepdim=True) / (y.shape[1] * y.shape[2])

        y = torch.nan_to_num(y, nan=0)
        loss_ = F.mse_loss(
            pred_score[mask], y[mask]
        )  # + F.mse_loss(coverage, pred_coverage) * coverage_weight
        pred_score, y = pred_score.reshape((len(pred_score), -1)), y.reshape((len(y), -1))
        val_loss[0] += loss_.item()
        corr = batch_pearson_correlation(pred_score, y).detach().cpu()[:, None]
        mean_pred_score += pred_score.mean() * len(pred_score)
        mean_y += y.mean() * len(y)
        mean_pred_coverage += pred_coverage.mean() * len(pred_coverage)
        mean_coverage += coverage.mean() * len(coverage)
        total_len += len(pred_score)

        across_batch_pearson[0].append(pred_score.detach().cpu().reshape((-1)))
        across_batch_pearson[1].append(y.detach().cpu().reshape((-1)))
        across_batch_pearson_coverage[0].append(coverage.detach().cpu().reshape((-1)))
        across_batch_pearson_coverage[1].append(pred_coverage.detach().cpu().reshape((-1)))
        profile_pearson.append(corr)

        size += 1
        bar.update(1)
        if validation_size is not None and size > validation_size:
            break

    val_loss = [l / size for l in val_loss]
    if len(profile_pearson) > 0:
        profile_pearson = torch.cat(profile_pearson, dim=0).mean(dim=0).detach().cpu().numpy()
    else:
        profile_pearson = np.array([0])

    pred_score, y = (
        torch.cat(across_batch_pearson[0], dim=0),
        torch.cat(across_batch_pearson[1], dim=0),
    )
    pred_coverage, coverage = (
        torch.cat(across_batch_pearson_coverage[1], dim=0),
        torch.cat(across_batch_pearson_coverage[0], dim=0),
    )

    mean_pred_score /= total_len
    mean_y /= total_len
    mean_pred_coverage /= total_len
    mean_coverage /= total_len
    if linear_correction:
        model = LinearRegression()

        a, b = pred_score.cpu().numpy().reshape((-1)), y.cpu().numpy().reshape((-1))
        if len(a) >= 1000000:
            idx = np.random.choice(len(a), 1000000, replace=False)
            a = a[idx]
            b = b[idx]
        mask = (~np.isnan(a)) & (~np.isnan(b)) & (~np.isinf(a)) & (~np.isinf(b))
        a, b = a[mask], b[mask]
        print(a.min(), a.max(), b.min(), b.max())
        model.fit(a.reshape((-1, 1)), b.reshape((-1, 1)))
        weight, bias = model.coef_[0, 0], model.intercept_[0]
        print("Linear correction", weight, bias)
    across_corr = [
        pearson_correlation(pred_score, y, mean_pred_score, mean_y),
        pearson_correlation(pred_coverage, coverage, mean_pred_coverage, mean_coverage),
    ]
    if not linear_correction:
        return val_loss, profile_pearson, across_corr
    else:
        return val_loss, profile_pearson, across_corr, weight, bias


class scFootprintBPNet(nn.Module):
    """
    This defines the bindingnet model
    """

    def __init__(
        self,
        dna_cnn_model=None,
        hidden_layer_model=None,
        profile_cnn_model=None,
        dna_len=2114,
        output_len=1000,
        embeddings=None,
        rank=8,
        hidden_dim=None,
        lora_dna_cnn=False,
        lora_dilated_cnn=False,
        lora_pff_cnn=False,
        lora_output_cnn=False,
        lora_count_cnn=False,
        method="lora",
        n_lora_layers=0,
        coverage=True,
    ):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.dna_len = dna_len
        self.output_len = output_len

        if embeddings is not None:
            if coverage:
                coverages = embeddings[:, -1][:, None]
                embeddings = embeddings[:, :-1]
            else:
                coverages = None

        if embeddings is not None:
            self.embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            self.embeddings.weight.data = torch.from_numpy(embeddings).float()
            self.embeddings.weight.requires_grad = False
            if coverage:
                self.coverages = (
                    nn.Embedding(coverages.shape[0], coverages.shape[1])
                    if coverages is not None
                    else None
                )
                self.coverages.weight.data = torch.from_numpy(coverages).float()
                self.coverages.weight.requires_grad = False
            else:
                self.coverages = None
        else:
            self.embeddings = None
            self.coverages = None

        if method == "lora":
            wrapper = Conv1dLoRAv2
        else:
            wrapper = BiasInjection

        if lora_dna_cnn:
            self.dna_cnn_model.conv = wrapper(
                self.dna_cnn_model.conv,
                A_embedding=self.embeddings,
                B_embedding=self.embeddings,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )

        hidden_layers = self.hidden_layer_model.layers
        for i in range(len(hidden_layers)):
            if lora_dilated_cnn:
                hidden_layers[i].module.conv1 = wrapper(
                    hidden_layers[i].module.conv1,
                    A_embedding=self.embeddings,
                    B_embedding=self.embeddings,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )
            if lora_pff_cnn:
                hidden_layers[i].module.conv2 = wrapper(
                    hidden_layers[i].module.conv2,
                    A_embedding=self.embeddings,
                    B_embedding=self.embeddings,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )

        if lora_output_cnn:
            self.profile_cnn_model.conv_layer = wrapper(
                self.profile_cnn_model.conv_layer,
                A_embedding=self.embeddings,
                B_embedding=self.embeddings,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )
        if isinstance(self.profile_cnn_model.linear, nn.Linear):
            print("translating linear into conv1d")
            weight = self.profile_cnn_model.linear.weight.data
            print(weight.shape)
            bias = self.profile_cnn_model.linear.bias.data
            self.profile_cnn_model.linear = Conv1dWrapper(weight.shape[1], weight.shape[0], 1)
            print(self.profile_cnn_model.linear.conv.weight.shape)
            self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            self.profile_cnn_model.linear.conv.bias.data = bias

        if lora_count_cnn:
            self.profile_cnn_model.linear = wrapper(
                self.profile_cnn_model.linear,
                A_embedding=self.embeddings,
                B_embedding=self.embeddings,
                r=1,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )
        if self.coverages is not None:
            self.profile_cnn_model = CoverageAdjustedFootprints_head(
                self.profile_cnn_model, self.coverages
            )

    def return_origin(self):
        self = self.to("cpu")
        # if isinstance(self.profile_cnn_model.linear, nn.Linear):
        #     print("translating linear into conv1d")
        #     weight = self.profile_cnn_model.linear.weight.data
        #     print(weight.shape)
        #     bias = self.profile_cnn_model.linear.bias.data
        #     self.profile_cnn_model.linear = Conv1dWrapper(weight.shape[1], weight.shape[0], 1)
        #     print(self.profile_cnn_model.linear.conv.weight.shape)
        #     self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
        #     self.profile_cnn_model.linear.conv.bias.data = bias

        model_clone = deepcopy(self)
        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = model_clone.dna_cnn_model.conv.layer
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv1, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv1 = layer.module.conv1.layer
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv2, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv2 = layer.module.conv2.layer

        if isinstance(model_clone.profile_cnn_model, CoverageAdjustedFootprints_head):
            model_clone.profile_cnn_model = model_clone.profile_cnn_model.footprints_head

        if not isinstance(model_clone.profile_cnn_model.conv_layer, Conv1dWrapper):
            model_clone.profile_cnn_model.conv_layer = (
                model_clone.profile_cnn_model.conv_layer.layer
            )
        if not isinstance(model_clone.profile_cnn_model.linear, Conv1dWrapper):
            model_clone.profile_cnn_model.linear = model_clone.profile_cnn_model.linear.layer

        return model_clone

    def collapse(self, cell, turn_on_grads=True):
        # if isinstance(self.profile_cnn_model.linear, nn.Linear):
        #     print("translating linear into conv1d")
        #     weight = self.profile_cnn_model.linear.weight.data
        #     print(weight.shape)
        #     bias = self.profile_cnn_model.linear.bias.data
        #     self.profile_cnn_model.linear = Conv1dWrapper(weight.shape[1], weight.shape[0], 1)
        #     print(self.profile_cnn_model.linear.conv.weight.shape)
        #     self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
        #     self.profile_cnn_model.linear.conv.bias.data = bias

        # self = self.to('cpu')
        model_clone = deepcopy(self)
        if not isinstance(model_clone.dna_cnn_model.conv, Conv1dWrapper):
            model_clone.dna_cnn_model.conv = model_clone.dna_cnn_model.conv.collapse_layer(cell)
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv1, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv1 = layer.module.conv1.collapse_layer(cell)
        if not isinstance(model_clone.hidden_layer_model.layers[0].module.conv2, Conv1dWrapper):
            for layer in model_clone.hidden_layer_model.layers:
                layer.module.conv2 = layer.module.conv2.collapse_layer(cell)
        if isinstance(model_clone.profile_cnn_model, CoverageAdjustedFootprints_head):
            # print ("collapse coverage adjusted")
            model = model_clone.profile_cnn_model.footprints_head
            model_clone.profile_cnn_model.collapse_layer(cell)
        else:
            model = model_clone.profile_cnn_model

        if not isinstance(model.conv_layer, Conv1dWrapper):
            model.conv_layer = model.conv_layer.collapse_layer(cell)
        if not isinstance(model.linear, Conv1dWrapper):
            model.linear = model.linear.collapse_layer(cell)
        if turn_on_grads:
            for p in model_clone.parameters():
                p.requires_grad = True

        return model_clone

    def forward(self, X, cells=None, output_len=None, modes=None):
        if output_len is None:
            output_len = self.output_len
        # get the motifs!
        X = self.dna_cnn_model(X, cells=cells)

        # get the hidden layer
        X = self.hidden_layer_model(X, cells=cells)

        # get the profile
        score = self.profile_cnn_model(X, cells=cells, output_len=output_len, modes=modes)

        return score

    def load_train_state_dict(self, ema, optimizer, scaler, savename):
        print("Nan training loss, load last OK-ish checkpoint")
        device = next(self.parameters()).device
        print(device)
        # self.cpu()
        checkpoint = torch.load(savename)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        # self.to(device)
        # optimizer = optimizer.to(device)
        # scaler = scaler.to(device)

        del checkpoint
        torch.cuda.empty_cache()
        # if ema:
        #     # ema = ema.cpu()
        #     ema = torch.load(savename + '.ema.pt')
        # ema = ema.to(device)
        return ema, optimizer, scaler

    def fit(
        self,
        dispmodel,
        training_data,
        validation_data=None,
        validation_size=None,
        max_epochs=100,
        optimizer=None,
        scheduler=None,
        validation_freq=100,
        early_stopping=False,
        return_best=True,
        savename="model",
        modes=np.arange(2, 101, 1),
        downsample=None,
        ema=None,
        use_amp=False,
        use_wandb=False,
        accumulate_grad=1,
        batch_size=None,
        coverage_warming=0,
    ):
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
        # -------

        """
        batch_size = training_data.batch_size if batch_size is None else batch_size
        early_stopping_counter = 0
        best_loss = np.inf
        index_all = list(np.arange(2, 101, 1))
        select_index = None
        random_modes = modes
        device = next(self.parameters()).device
        loss_history = []
        assert (
            validation_size is not None or validation_data is not None
        ), "Either validation_size or validation_data should be provided"

        if validation_data is None:
            validation_data = training_data
        if validation_size is None:
            validation_size = len(validation_data)
        if validation_freq is None:
            validation_freq = len(training_data)
        if use_amp:
            print("Using amp")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        # min_scale = 128
        if coverage_warming > 0:
            # backup = self.embeddings.weight.data.clone()
            # only using coverage to finetune...
            # self.embeddings.weight.data[:] *= 0
            requires_grads = {}
            for name, param in self.named_parameters():
                if name in requires_grads:
                    print("duplicated name", name, param.requires_grad, requires_grads[name])
                requires_grads[name] = param.requires_grad
                param.requires_grad = False
            for p in self.profile_cnn_model.adjustment_count.parameters():
                p.requires_grad = True
            for p in self.profile_cnn_model.adjustment_footprint.parameters():
                p.requires_grad = True

        for epoch in range(max_epochs):
            bar = trange(
                validation_freq,
                desc=(
                    " - (Training) {epoch}".format(epoch=epoch + 1) + " warming"
                    if (coverage_warming > 0 and epoch <= coverage_warming)
                    else ""
                ),
                leave=False,
                dynamic_ncols=True,
            )
            moving_avg_loss = 0
            iteration = 0

            if epoch > coverage_warming and coverage_warming > 0:
                for name, param in self.named_parameters():
                    param.requires_grad = requires_grads[name]
                for p in self.profile_cnn_model.adjustment_count.parameters():
                    p.requires_grad = False
                for p in self.profile_cnn_model.adjustment_footprint.parameters():
                    p.requires_grad = False

            training_index_all = np.random.permutation(len(training_data))
            training_data_epoch_loader = training_data.resample()
            for data in training_data_epoch_loader:
                try:
                    autocast_context = torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                    )
                except RuntimeError:
                    autocast_context = torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=use_amp
                    )
                with autocast_context:
                    random_modes = np.random.permutation(modes)[:30]
                    select_index = torch.as_tensor([index_all.index(mode) for mode in random_modes])
                    if len(data) == 2:
                        X, y = data
                        cell = None
                        norm_cov = None
                    else:
                        X, y, cell, peak, norm_cov = data
                        X = X[:batch_size]
                        y = y[:batch_size]
                        cell = cell[:batch_size, 0]
                        cell = cell.to(device)
                        # norm_cov = norm_cov[:batch_size].to(device)
                        norm_cov = None

                    X = X.to(device)
                    y = y.to(device)

                    atac = y[:, 0]
                    if downsample is not None:
                        atac = F.dropout(atac, 1 - downsample, training=self.training)

                    footprints = multiscaleFoot(atac, y[:, 1], random_modes, dispmodel)
                    mask = ~torch.isnan(footprints)
                    # footprints = footprints - torch.nansum(footprints, dim=(1,2), keepdim=True) / (footprints.shape[1] * footprints.shape[2])
                    if norm_cov is not None:
                        coverage = norm_cov
                    else:
                        coverage = y[:, 0].sum(dim=-1)
                        # footprints[coverage < 10] = torch.nan
                        coverage = torch.log1p(coverage)

                    pred_score, pred_coverage = self.forward(X, cell, modes=select_index)

                    desc_str = " - (Training) {epoch}".format(epoch=epoch + 1)

                    loss_footprint = F.mse_loss(pred_score[mask], footprints[mask])
                    desc_str += " Footprint Loss: {loss:.2f}".format(loss=loss_footprint.item())

                    loss_coverage = F.mse_loss(coverage, pred_coverage)
                    desc_str += " Coverage Loss: {loss:.2f}".format(loss=loss_coverage.item())
                    desc_str += (
                        " warming" if (coverage_warming > 0 and epoch <= coverage_warming) else ""
                    )
                    loss = (loss_footprint + loss_coverage) / accumulate_grad
                    if np.isnan(loss.item()):
                        ema, optimizer, scaler = self.load_train_state_dict(
                            ema, optimizer, scaler, savename
                        )
                        continue

                scaler.scale(loss).backward()
                moving_avg_loss += loss_footprint.item()
                if (iteration + 1) % accumulate_grad == 0:
                    scaler.unscale_(
                        optimizer
                    )  # Unscale gradients for clipping without inf/nan gradients affecting the model
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)  # Adjust max_norm accordingly

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # if scaler._scale < min_scale:
                    #     scaler._scale = torch.tensor(min_scale).to(scaler._scale)

                    if ema:
                        ema.update()
                    if scheduler is not None:
                        scheduler.step()
                bar.set_description(desc_str)
                bar.update(1)
                iteration += 1
                if iteration >= validation_freq:
                    break

            print(
                " - (Training) {epoch} Loss: {loss:.2f}".format(
                    epoch=epoch + 1, loss=moving_avg_loss / iteration
                )
            )
            print("Learning rate", optimizer.param_groups[0]["lr"])

            bar.close()
            self.eval()

            val_loss, profile_pearson, across_pearson = validation_step_footprint(
                self, validation_data, validation_size, dispmodel, modes
            )

            val_loss_all = np.sum(val_loss)
            if np.isnan(val_loss_all):
                print("Nan loss, load last OK-ish checkpoint")
                ema, optimizer, scaler = self.load_train_state_dict(
                    ema, optimizer, scaler, savename
                )
                # optimizer.load_state_dict(torch.load(savename)['optimizer'])
                # wandb.log({"train/train_loss": np.nan,
                #             "val/val_loss": np.nan,
                #             "val/best_val_loss": np.nan,
                #             "val/profile_pearson": np.nan,
                #             "val/across_pearson_footprint": np.nan,
                #             "val/across_pearson_coverage": np.nan,
                #            "epoch": epoch})

                # break
            print(
                " - (Validation) {epoch} \
                        Loss: {loss:.5f}".format(
                    epoch=epoch + 1, loss=val_loss_all
                )
            )
            print("Profile pearson", profile_pearson)
            print("Across peak pearson", across_pearson)

            if ema:
                ema.eval()
                ema.ema_model.eval()
                val_loss, profile_pearson, across_pearson = validation_step_footprint(
                    ema.ema_model, validation_data, validation_size, dispmodel, modes
                )
                if np.sum(val_loss) > val_loss_all:
                    # ema not converged yet:
                    early_stopping_counter = 0

                val_loss_all = np.sum(val_loss)

                if np.isnan(val_loss_all):
                    print("Nan loss, load last OK-ish checkpoint")
                    ema, optimizer, scaler = self.load_train_state_dict(
                        ema, optimizer, scaler, savename
                    )
                    # print("Nan loss")
                    # wandb.log({"train/train_loss": np.nan,
                    #            "val/val_loss": np.nan,
                    #            "val/best_val_loss": np.nan,
                    #            "val/profile_pearson": np.nan,
                    #            "val/across_pearson_footprint": np.nan,
                    #            "val/across_pearson_coverage": np.nan,
                    #            "epoch": epoch})
                    # break
                print(
                    " - (Validation) {epoch} \
                Loss: {loss:.5f}".format(
                        epoch=epoch + 1, loss=val_loss_all
                    )
                )
                print("EMA Profile pearson", profile_pearson)
                print("EMA Across peak pearson", across_pearson)
                ema.train()

            self.train()

            loss_history.append([moving_avg_loss / iteration, val_loss_all])

            if val_loss_all < best_loss:
                print("best loss", val_loss_all)
                best_loss = val_loss_all
                early_stopping_counter = 0
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema else None,
                }
                torch.save(checkpoint, savename)
                torch.save(self, savename + ".model.pt")
                if ema:
                    torch.save(ema, savename + ".ema.pt")
                    torch.save(ema.ema_model, savename + ".ema_model.pt")
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    print("Early stopping")
                    break

            if use_wandb:
                wandb.log(
                    {
                        "train/train_loss": moving_avg_loss / iteration,
                        "val/val_loss": val_loss_all,
                        "val/best_val_loss": best_loss,
                        "val/profile_pearson": profile_pearson,
                        "val/across_pearson_footprint": across_pearson[0],
                        "val/across_pearson_coverage": across_pearson[1],
                        "epoch": epoch,
                    }
                )

        if return_best:
            self.load_state_dict(torch.load(savename)["state_dict"])
            print("loaded best model")

        return epoch, loss_history

    def predict(self, *args, **kwargs):
        return predict(self, *args, **kwargs)
