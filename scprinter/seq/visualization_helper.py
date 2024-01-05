import torch
import numpy as np
from tqdm.auto import tqdm
from ..utils import regionparser



def predict_footprints(model,
                          printer,
                          regions,
                          modes=np.arange(2,101,1),
                          verbose=False,
                          batch_size=64):
    """

    Parameters
    ----------
    model
    summits
    modes

    Returns
    -------

    """
    dev = next(model.parameters()).device
    modes_all = list(np.arange(2, 101, 1))
    modes = [modes_all.index(x) for x in modes]
    regions = regionparser(regions, printer, model.dna_len)
    X = torch.stack([printer.genome.fetch_onehot_seq(region[0],
                                            region[1],
                                            region[2]) for region in np.array(regions)], dim=0)

    starts = np.arange(0, X.shape[0], batch_size)
    footprints = []
    for start in tqdm(starts, disable=not verbose):
        end = min(start + batch_size, X.shape[0])
        X_batch = X[start:end]
        with torch.no_grad():
            pred_footprint = model(X_batch.float().to(dev), modes=modes)[0].detach().cpu().numpy()
        footprints.append(pred_footprint)
    footprints = np.concatenate(footprints, axis=0)
    return footprints


