#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""Robustness application

CIFAR-10 with epsilon-adversarial loss constraint

"""

import foolbox

import torch
import torchvision
import torch.nn.functional as F

from resnet import ResNet18

import numpy as np

import copy

import sys, os
sys.path.append(os.path.abspath('../'))

import csl, csl.datasets

# Perturbation magnitude
eps = 0.02

# Use GPU if available
theDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


####################################
# FUNCTIONS                        #
####################################
def accuracy(yhat, y):
    _, predicted = torch.max(yhat, 1)
    correct = (predicted == y).sum().item()
    return correct/yhat.shape[0]

def preprocess(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype = img.dtype, device=theDevice).reshape((3, 1, 1))
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype = img.dtype, device=theDevice).reshape((3, 1, 1))
    return (img - mean) / std


####################################
# DATA                             #
####################################
n_train = 4900
n_valid = 100

target = csl.datasets.CIFAR10(root = 'data', train = True)[:][1]

label_idx = [np.flatnonzero(target == label) for label in range(0,10)]
label_idx = [np.random.RandomState(seed=42).permutation(idx) for idx in label_idx]
train_subset = [idx[:n_train] for idx in label_idx]
train_subset = np.array(train_subset).flatten()

train_transform = torchvision.transforms.Compose([
    csl.datasets.utils.RandomFlip(),
    csl.datasets.utils.RandomCrop(size=32,padding=4),
    csl.datasets.utils.ToTensor(device=theDevice)
    ])

trainset = csl.datasets.CIFAR10(root = 'data', train = True, subset = train_subset,
                                transform = train_transform,
                                target_transform = csl.datasets.utils.ToTensor(device=theDevice))

valid_subset = [idx[n_train:n_train+n_valid] for idx in label_idx]
valid_subset = np.array(valid_subset).flatten()
validset = csl.datasets.CIFAR10(root = 'data', train = True, subset = valid_subset,
                                transform = csl.datasets.utils.ToTensor(device=theDevice),
                                target_transform = csl.datasets.utils.ToTensor(device=theDevice))


####################################
# CONSTRAINED LEARNING PROBLEM     #
####################################
class robustLoss(csl.ConstrainedLearningProblem):
    def __init__(self, rhs):
        self.model = csl.PytorchModel(ResNet18().to(theDevice))
        self.data = trainset
        self.batch_size = 256

        self.obj_function = self.obj_fun

        # Constraints
        self.constraints = [self.adversarialLoss]
        self.rhs = [rhs]

        self.foolbox_model = foolbox.PyTorchModel(self.model.model, bounds=(0, 1),
                                                  device=theDevice,
                                                  preprocessing = dict(mean=[0.4914, 0.4822, 0.4465],
                                                                       std=[0.2023, 0.1994, 0.2010],
                                                                       axis=-3))
        self.attack = foolbox.attacks.LinfPGD(rel_stepsize = 1/3, abs_stepsize = None,
                                              steps = 5, random_start = True)

        super().__init__()

    def obj_fun(self, batch_idx):
        x, y = self.data[batch_idx]

        yhat = self.model(preprocess(x))

        return 0.1*self._loss(yhat, y)

    def adversarialLoss(self, batch_idx, primal):
        x, y = self.data[batch_idx]

        # Attack
        self.model.eval()

        # Save gradients before adversarial runs
        saved_grad = [copy.deepcopy(p.grad) for p in self.model.parameters]

        # Dual is computed in a no_grad() environment
        x_processed, _, _ = self.attack(self.foolbox_model, x, y, epsilons = eps)

        # Reload gradients
        for p,g in zip(self.model.parameters, saved_grad):
            p.grad = g

        if primal:
            self.model.train()
            yhat = self.model(preprocess(x_processed))
            loss = self._loss(yhat, y)
        else:
            with torch.no_grad():
                yhat = self.model(preprocess(x_processed))
                loss = self._loss(yhat, y)
            self.model.train()

        return loss

    @staticmethod
    def _loss(yhat, y):
        return F.cross_entropy(yhat, y)

####################################
# TRAINING                         #
####################################
def validation_hook(problem, solver_state):
        adv_epoch = 10
        _adv_epoch = adv_epoch

        batch_idx = np.arange(0, len(validset)+1, problem.batch_size)
        if batch_idx[-1] < len(validset):
            batch_idx = np.append(batch_idx, len(validset))

        # Validate
        acc = 0
        acc_adv = 0
        problem.model.eval()
        for batch_start, batch_end in zip(batch_idx, batch_idx[1:]):
            x, y = validset[batch_start:batch_end]
            with torch.no_grad():
                yhat = problem.model(preprocess(x))
                acc += accuracy(yhat, y)*(batch_end - batch_start)/len(validset)

            # Attack
            if _adv_epoch == 1:
                adversarial, _, _ = problem.attack(problem.foolbox_model, x, y, epsilons = eps)
                with torch.no_grad():
                    yhat_adv = problem.model(preprocess(adversarial))
                    acc_adv += accuracy(yhat_adv, y)*(batch_end - batch_start)/len(validset)
        problem.model.train()

        # Results
        if _adv_epoch > 1:
            print(f"Validation accuracy: {acc*100:.2f} / Dual variables: {[lambda_value.item() for lambda_value in problem.lambdas]}")
            _adv_epoch -= 1
        else:
            print(f"Validation accuracy:{acc*100:.2f} / Adversarial accuracy = {acc_adv*100:.2f}")
            _adv_epoch = adv_epoch

        return False

problem = robustLoss(rhs=0.7)

solver_settings = {'iterations': 400,
                   'verbose': 1,
                   'batch_size': 128,
                   'primal_solver': lambda p: torch.optim.Adam(p, lr=0.01),
                   'lr_p_scheduler': None,
                   'dual_solver': lambda p: torch.optim.Adam(p, lr=0.001),
                   'lr_d_scheduler': None,
                   'device': theDevice,
                   'STOP_USER_DEFINED': validation_hook,
                   }
solver = csl.SimultaneousPrimalDual(solver_settings)

solver.solve(problem)
solver.plot()


####################################
# TESTING                          #
####################################
# Test data
testset = csl.datasets.CIFAR10(root = 'data', train = False,
                               transform = csl.datasets.utils.ToTensor(device=theDevice),
                               target_transform = csl.datasets.utils.ToTensor(device=theDevice))

# Adversarial attack
problem.model.eval()
foolbox_model = foolbox.PyTorchModel(problem.model.model, bounds=(0, 1),
                                     device=theDevice,
                                     preprocessing = dict(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010],
                                                          axis=-3))
attack = foolbox.attacks.LinfPGD(rel_stepsize = 1/30, abs_stepsize = None,
                                 steps = 50, random_start = True)
epsilon_test = np.linspace(0.01,0.06,7)

# Prepare batches
batch_idx = np.arange(0, len(testset)+1, problem.batch_size)
if batch_idx[-1] < len(testset):
    batch_idx = np.append(batch_idx, len(testset))

n_total = 0
acc_test = 0
acc_adv = np.zeros(epsilon_test.shape[0])
success_adv = np.zeros_like(acc_adv)

for batch_start, batch_end in zip(batch_idx, batch_idx[1:]):
    x_test, y_test = testset[batch_start:batch_end]

    # Nominal accuracy
    yhat = problem.model(preprocess(x_test))
    acc_test += accuracy(yhat, y_test)*(batch_end - batch_start)

    # Adversarials accuracy
    adversarials, _, success = attack(foolbox_model, x_test, y_test, epsilons = epsilon_test)
    for ii, adv in enumerate(adversarials):
        yhat_adv = problem.model(preprocess(adv))
        acc_adv[ii] += accuracy(yhat_adv, y_test)*(batch_end - batch_start)
        success_adv[ii] += torch.sum(success[ii])

    n_total += batch_end - batch_start

acc_test /= n_total
acc_adv /= n_total
success_adv /= n_total

print('====== TEST ======')
print(f'Test accuracy: {100*acc_test:.2f}')
print(f'Adversarial accuracy: {100*acc_adv}')
print(f'Adversarial success: {100*success_adv}')
