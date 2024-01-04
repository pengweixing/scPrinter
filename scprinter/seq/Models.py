import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from .Functions import *
from .minimum_footprint import *
from scipy.stats import pearsonr
import wandb

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
            X_batch = X[i:i + batch_size].to(device)
            X_foot, X_score = self(X_batch)
            pred_footprints.append(X_foot.detach().cpu())
            pred_scores.append(X_score.detach().cpu())
        pred_footprints = torch.cat(pred_footprints, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)
    return pred_footprints, pred_scores

@torch.no_grad()
def validation_step_footprint(model,
                    validation_data,
                    validation_size,
                    dispmodel,
                    modes,
                    coverage_weight=1.0,
                    verbose=False):
    device = next(model.parameters()).device
    size = 0
    profile_pearson = []
    across_batch_pearson = [[], []]
    across_batch_pearson_coverage = [[], []]
    val_loss = [0]
    for data in tqdm(validation_data, disable=not verbose, dynamic_ncols=True):
        X, y = data
        pred_score, pred_coverage = predict(model, X, batch_size=X.shape[0])
        pred_score = pred_score.to(device)
        pred_coverage = pred_coverage.to(device)
        y = y.to(device)
        coverage = torch.log1p(y[:, 0].sum(dim=-1))
        y = multiscaleFoot(y[:, 0],
                            y[:, 1],
                            modes,
                            dispmodel)
        mask = ~torch.isnan(y)
        y = torch.nan_to_num(y, nan=0)
        loss_ = F.mse_loss(pred_score[mask], y[mask]) #+ F.mse_loss(coverage, pred_coverage) * coverage_weight
        pred_score, y = pred_score.reshape((len(pred_score), -1)), y.reshape((len(y), -1))
        val_loss[0] += loss_.item()
        corr = batch_pearson_correlation(pred_score, y).detach().cpu()[:, None]
        across_batch_pearson[0].append(pred_score.detach().cpu().reshape((-1)))
        across_batch_pearson[1].append(y.detach().cpu().reshape((-1)))
        across_batch_pearson_coverage[0].append(coverage.detach().cpu().reshape((-1)))
        across_batch_pearson_coverage[1].append(pred_coverage.detach().cpu().reshape((-1)))
        profile_pearson.append(corr)

        size += 1
        if validation_size is not None and size > validation_size:
            break

    val_loss = [l / size for l in val_loss]
    if len(profile_pearson) > 0:
        profile_pearson = torch.cat(profile_pearson, dim=0).mean(dim=0).detach().cpu().numpy()
    else:
        profile_pearson = np.array([0])
    pred_score, y = (torch.cat(across_batch_pearson[0], dim=0)[None],
                     torch.cat(across_batch_pearson[1], dim=0)[None])
    pred_coverage, coverage = (torch.cat(across_batch_pearson_coverage[0], dim=0)[None],
                               torch.cat(across_batch_pearson_coverage[1], dim=0)[None])
    across_corr = [batch_pearson_correlation(pred_score, y)[0],
                  batch_pearson_correlation(pred_coverage, coverage)[0]]
    return val_loss, profile_pearson, across_corr

@torch.no_grad()
def downsample_atac(atac, downsample_rate):
    n_remove_insertions = int(torch.sum(atac > 0) * (1 - downsample_rate))
    non_zero_place = torch.where(atac > 0)
    reduce = np.random.choice(len(non_zero_place[0]),
                              n_remove_insertions,
                              replace=False)
    atac[non_zero_place[0][reduce], non_zero_place[1][reduce]] = 0
    return atac

class FootprintBPNet(nn.Module):
    """
    This defines the bindingnet model
    """

    def __init__(self,
                 dna_cnn_model=None,
                 hidden_layer_model=None,
                 profile_cnn_model=None,
                 dna_len=2114,
                 output_len=1000):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.dna_len = dna_len
        self.output_len = output_len
        self.upsample=True

    def forward(self, X, output_len=None, modes=None):
        if output_len is None:
            output_len = self.output_len
        # get the motifs!
        # print (X.shape, output_len)
        X = self.dna_cnn_model(X)

        # get the hidden layer
        X = self.hidden_layer_model(X)

        # get the profile
        score = self.profile_cnn_model(X,
                                       output_len=output_len,
                                       upsample=self.upsample,
                                       modes=modes)

        return score

    def fit(self,
            dispmodel,
            training_data,
            validation_data=None,
            validation_size=None,
            max_epochs=100,
            optimizer=None,
            scheduler=None,
            validation_freq=100,
            early_stopping=False,
            return_best=True, savename='model',
            modes=np.arange(2,101,1),
            coverage_weight=1.0,
            downsample=None,
            ema=None,
            use_amp=False,
            use_wandb=False):
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
        index_all = list(np.arange(2, 101, 1))
        # select_index = torch.as_tensor([index_all.index(mode) for mode in modes])
        select_index = None
        random_modes = modes
        device = next(self.parameters()).device
        loss_history = []
        assert validation_size is not None or validation_data is not None, "Either validation_size or validation_data should be provided"

        if validation_data is None:
            validation_data = training_data
        if validation_size is None:
            validation_size = len(validation_data)
        if validation_freq is None:
            validation_freq = len(training_data)
        if use_amp:
            print ("Using amp")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(max_epochs):
            bar = trange(validation_freq, desc=' - (Training) {epoch}'.format(
                epoch=epoch + 1), leave=False, dynamic_ncols=True)
            moving_avg_loss = 0
            iteration = 0
            training_data_epoch_loader = training_data.resample()
            calib_ratio = True
            for data in training_data_epoch_loader:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    random_modes = np.random.permutation(modes)[:30]
                    select_index = torch.as_tensor([index_all.index(mode) for mode in random_modes])

                    X, y = data
                    X = X.to(device)
                    y = y.to(device)
                    atac = y[:, 0]
                    if downsample is not None:
                        atac = F.dropout(atac, 1-downsample, training=self.training)
                        # atac = downsample_atac(atac, downsample)



                    footprints = multiscaleFoot(atac,
                                                y[:, 1],
                                                random_modes,
                                                dispmodel)
                    coverage = torch.log1p(y[:, 0].sum(dim=-1))

                    optimizer.zero_grad()
                    pred_score, pred_coverage = self.forward(X, modes=select_index)

                    loss = 0
                    desc_str = ' - (Training) {epoch}'.format(
                        epoch=epoch + 1)
                    # footprints = torch.zeros_like(pred_score).cuda()
                    mask = ~torch.isnan(footprints)
                    loss_footprint = F.mse_loss(pred_score[mask], footprints[mask])
                    desc_str += " Footprint Loss: {loss:.2f}".format(loss=loss_footprint.item())

                    loss_coverage = F.mse_loss(coverage,
                                       pred_coverage)
                    desc_str += " Coverage Loss: {loss:.2f}".format(loss=loss_coverage.item())

                    # if calib_ratio:
                    #     optimizer.zero_grad(set_to_none=True)
                    #     loss_footprint.backward(retain_graph=True)
                    #     norm1 = self.dna_cnn_model.conv.weight.grad.data.norm(2)
                    #     optimizer.zero_grad(set_to_none=True)
                    #     loss_coverage.backward(retain_graph=True)
                    #     norm2 = self.dna_cnn_model.conv.weight.grad.data.norm(2)
                    #     ratio = coverage_weight * norm1 / norm2
                    #     ratio = min(ratio, coverage_weight)
                    #     optimizer.zero_grad()
                    #     calib_ratio = False

                    loss = loss_footprint + loss_coverage #* ratio
                    # desc_str += " Ratio: {loss:.2f}".format(loss=ratio)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()
                    if ema:
                        ema.update()
                    if scheduler is not None:
                        scheduler.step()
                    bar.set_description(desc_str)
                    bar.update(1)
                    iteration += 1
                    moving_avg_loss += loss.item()
                    if iteration >= validation_freq:
                        break

            print(' - (Training) {epoch} Loss: {loss:.2f}'.format(
                    epoch=epoch + 1,
                    loss=moving_avg_loss / iteration))
            print ("Learning rate", optimizer.param_groups[0]['lr'])

            bar.close()
            self.eval()

            val_loss, profile_pearson, across_pearson = validation_step_footprint(self,
                                                                                  validation_data,
                                                                                  validation_size,
                                                                                  dispmodel,
                                                                                  modes,
                                                                                  coverage_weight,
                                                                                  verbose=True)

            val_loss_all = np.sum(val_loss)
            print(' - (Validation) {epoch} \
                        Loss: {loss:.5f}'.format(
                epoch=epoch + 1, loss=val_loss_all))
            print("Profile pearson", profile_pearson)
            print("Across peak pearson", across_pearson)

            if ema:
                val_loss, profile_pearson, across_pearson = validation_step_footprint(ema,
                                                           validation_data,
                                                           validation_size,
                                                             dispmodel,
                                                            modes,
                                                            coverage_weight,
                                                            verbose=True)

                val_loss_all = np.sum(val_loss)
                print(' - (Validation) {epoch} \
                Loss: {loss:.5f}'.format(
                    epoch=epoch + 1, loss=val_loss_all))
                print ("EMA Profile pearson", profile_pearson)
                print("EMA Across peak pearson", across_pearson)



            self.train()

            loss_history.append([moving_avg_loss/ iteration, val_loss_all])

            if val_loss_all < best_loss:
                print ("best loss", val_loss_all)
                best_loss = val_loss_all
                early_stopping_counter = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, savename)
                torch.save(self, savename + '.model.pt')
                if ema:
                    torch.save(ema, savename + '.ema.pt')
                    torch.save(ema.ema_model, savename + '.ema_model.pt')
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    print('Early stopping')
                    break

            if use_wandb:
                wandb.log({"train/train_loss": moving_avg_loss / iteration,
                           "val/val_loss": val_loss_all,
                           "val/best_val_loss": best_loss,
                           "val/profile_pearson": profile_pearson,
                           "val/across_pearson_footprint": across_pearson[0],
                           "val/across_pearson_coverage": across_pearson[1],
                           "epoch": epoch})

        if return_best:
            self.load_state_dict(torch.load(savename)['state_dict'])
            print ("loaded best model")

        return epoch, loss_history
    def predict(self, *args, **kwargs):
        return predict(self, *args, **kwargs)