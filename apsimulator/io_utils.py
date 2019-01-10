# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import os

def determine_project_root_path():
    source_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.split(source_path)[0]
    return root_path


project_root_path = determine_project_root_path()


def get_path_relative_to_root(*path_components):
    return os.path.join(project_root_path, *path_components)


def get_filename_base(path_of_file):
    return os.path.splitext(os.path.split(path_of_file)[1])[0]
