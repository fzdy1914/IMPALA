"""Learner with parameter server"""
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange

import vtrace
from utils import make_time_major


def learner(model, data, ps, args):
    """Learner to get trajectories from Actors."""
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    batch_size = args.batch_size
    baseline_cost = args.baseline_cost
    entropy_cost = args.entropy_cost
    gamma = args.gamma
    save_path = args.save_path
    load_path = args.load_path
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print("Loaded model from:", load_path)
    else:
        print("Model %s do not exist" % load_path)

    """Gets trajectories from actors and trains learner."""
    batch = []
    t = trange(10000000)
    for epoch in t:
        trajectory = data.get()
        batch.append(trajectory)
        if torch.cuda.is_available():
            trajectory.cuda()
        if len(batch) < batch_size:
            continue
        boards, actions, rewards, dones, behaviour_logits = make_time_major(batch)
        optimizer.zero_grad()
        logits, values = model(boards)
        bootstrap_value = values[-1]
        actions, rewards, dones, behaviour_logits = actions[1:], rewards[1:], dones[1:], behaviour_logits[1:]
        logits, values = logits[:-1], values[:-1]
        discounts = (~dones).float() * gamma
        vs, pg_advantages = vtrace.from_logits(
            behaviour_policy_logits=behaviour_logits,
            target_policy_logits=logits,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value)
        # policy gradient loss
        cross_entropy = F.cross_entropy(logits, actions, reduction='none')
        loss = (cross_entropy * pg_advantages.detach()).sum()
        # baseline_loss
        loss += baseline_cost * .5 * (vs - values).pow(2).sum()
        # entropy_loss
        loss += entropy_cost * -(-F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(-1).sum()
        loss.backward()
        optimizer.step()
        model_state = dict()
        for name, tensor in model.state_dict(keep_vars=True).items():
            model_state[name] = tensor.detach().clone().cpu()
        ps.push(model_state)
        if epoch % 10000 == 0:
            print("save model: %s" % epoch)
            torch.save(model.state_dict(), save_path % epoch)
        batch = []
