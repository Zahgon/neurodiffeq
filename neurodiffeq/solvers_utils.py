import os
import json
import dill
import numpy as np
import pathlib
import torch
import requests
import tempfile
from typing import Union
from itertools import chain
import inspect
import ast
import types
import random

try:
    DEV = os.environ["DEV"]
    DEV = True
except KeyError:
    DEV = False

try:
    NEURODIFF_API_URL = os.environ["NEURODIFF_API_URL"]
except KeyError:
    if DEV:
        NEURODIFF_API_URL = "http://dev.neurodiff.io/api/v1"
    else:
        NEURODIFF_API_URL = "http://www.neurodiff.io/api/v1"


def is_solution_name(name):
    pass


def process_response(response):
    pass


def _make_api_headers():
    pass


def create_cache_dir():
    pass


def get_file(url, name):
    pass


def get_source(lambda_function):
    pass


def get_parameters(lambda_function):
    pass


def get_conditions(conditions):
    pass


def get_generator(generator):
    pass


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        pass


def get_sample_solution1D(solver):
    pass


def get_sample_solution2D(solver):
    pass

def get_sample_solutionBundle1D(solver):
    pass

def get_networks(solver):
    pass


def get_loss(loss):
    pass


class SolverConfig():
    conditions = None
    ode_system = None
    pde_system = None
    nets = None
    best_nets = None
    optimizer = None
    optimizer_params = None
    train_generator = None
    valid_generator = None


class PretrainedSolver():
    diff_eqs_source = ""

    def print_diff_eqs(self):
        pass


    def save(self,
             path: str = None,
             name: str = None,
             save_to_hub=False):
        pass



    @classmethod
    def load(cls,
             path: str = None,
             name: str = None,
             config=SolverConfig()):
        pass
