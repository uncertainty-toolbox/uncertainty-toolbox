"""Touch up the conda recipe from grayskull using conda-souschef."""
import os
from shutil import copyfile
from os.path import join
from pathlib import Path

# from chardet import detect
from souschef.recipe import Recipe

import uncertainty_toolbox as module

os.system(f"grayskull pypi {module.__name__}=={module.__version__}")

fpath = join(module.__name__, "meta.yaml")
Path("scratch").mkdir(exist_ok=True)
fpath2 = join("scratch", "meta.yaml")
my_recipe = Recipe(load_file=fpath)
my_recipe["requirements"]["host"].append("flit")
del my_recipe["test"]["imports"].yaml[0]  # delete "import tests" test
my_recipe.save(fpath)
my_recipe.save(fpath2)

copyfile("LICENSE", join("scratch", "LICENSE"))
