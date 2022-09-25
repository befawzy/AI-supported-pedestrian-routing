import torch
import random


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    
    Parameters
    ----------
    input: PyTorch Tensor of shape (N, ) giving scores.
    
    target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns
    ----------

    A PyTorch Tensor containing the mean BCE loss over the minibatch of
    input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    calculate generator loss

    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    calculate discriminator loss

    Parameters
    ----------
    scores_real: Tensor of shape (N,) giving scores for real samples
    
    scores_fake: Tensor of shape (N,) giving scores for fake samples

    Returns
    ----------
    loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    calculate l2 loss

    Parameters
    ----------
    pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    
    pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    
    loss_mask: Tensor of shape (batch, seq_len)
    
    mode: Can be one of sum, average, raw
    
    Returns
    -------
    loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    calculate average displacement error, i.e. Average L2 distance between ground truth 
    and our prediction over all predicted time steps

    Parameters
    ----------
    pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    
    pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    
    consider_ped: Tensor of shape (batch)
    
    mode: Can be one of sum, raw
    
    Returns
    -------

    loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    calculate final displacement error, i.e. The distance between the predicted final
    destination and the true final destination at end of the prediction period.

    Parameters
    ----------
    
    pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    
    pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    
    consider_ped: Tensor of shape (batch)
    
    Returns
    ----------
    loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
