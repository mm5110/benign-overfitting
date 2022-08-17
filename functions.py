import torch
import numpy as np
from tqdm import tqdm


def train_model(writer, model, optimizer, loss_fn, train_x, train_y, test_x, test_y, num_epochs, num_corrupted, plot_incs, test_size, train_size, w_star):

    # Run SGD with fixed stepsize for specified number of epochs
    for epoch in tqdm(range(num_epochs+1)):

        # Compute generalization error at specified intervals
        if epoch in plot_incs:
            test_output = model(test_x)
            Wt = model.fc1.weight
            loss = loss_fn(test_output, test_y,Wt)
            writer.add_scalar("Test loss", loss.item(), epoch)
            writer.add_scalar("Test error", torch.sum(torch.abs(0.5 * (torch.sign(test_output) - test_y))).item()/test_size, epoch)
            W = (((model.fc1.weight).detach()).numpy())# returns m x d matrix
            U, D, Vt = np.linalg.svd(W, full_matrices=False)
            corrs = Vt@w_star
            weighted_corrs = D@Vt@w_star
            writer.add_histogram("SV correlation with target", corrs,epoch)
            writer.add_histogram("Weighted SV correlation with target", weighted_corrs, epoch)
            writer.add_histogram("Singular value distribution", D, epoch)
            sing_ratio = D[0]/np.sum(D)
            writer.add_scalar("Rank-one-ness ratio", sing_ratio,epoch)

        optimizer.zero_grad()
        output = model(train_x)
        Wt = model.fc1.weight
        loss = loss_fn(output, train_y, Wt)
        writer.add_scalar("Train error", torch.sum(torch.abs(0.5 * (torch.sign(output) - train_y))).item()/train_size, epoch)
        writer.add_scalar("Train loss", loss.item(), epoch)
        if num_corrupted > 0:
            Wt = model.fc1.weight
            writer.add_scalar("Train loss clean", loss_fn(output[num_corrupted:], train_y[num_corrupted:], Wt).item(), epoch)
            writer.add_scalar("Train loss corrupted", loss_fn(output[0:num_corrupted], train_y[0:num_corrupted], Wt).item(), epoch)

        loss.backward()
        optimizer.step()









