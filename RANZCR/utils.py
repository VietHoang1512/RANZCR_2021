import os
import random

import numpy as np
import pytorch_lightning as pl
import torch


def seed_everything(seed=1710):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.bench_mark = True  # for faster training but not deterministic
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


def print_signature():
    os.system("")
    print(
        "\033[36m"
        + r'''
                                        .
                         |              |
                         |             ,|
                ,_       |\            F'   ,.-""`.
               /  `-._   |`           // ,-"_,..  |
              |   ___ `. | \ ,"""`-. /.-' ,'    ' |_....._
              |  /   `-.`.  L......|j j_.'      ` |       `._
              | |      _,'| |______|' | '-._     ||  ,.-.    `.
               L|    ,'   | |      | j      `-.  || '    `.    \
___            | \_,'     | |`"----| |         `.||       |\    \
 ""=+...__     `,'   ,.-.   |....._|   _....     Y \      j_),..+=--
     `"-._"._  .   ,' |  `   \    /  ,' |   \     \ j,..-"_..+-"  L
          `-._-+. j   !   \   `--'  .   !    \  ,.-" _..<._  |    |
              |-. |   |    L        |   !     |  .-/'      `.|-.,-|
              |__ '   '    |        '   |    /    /|   `, -. |   j
        _..--'"__  `-.___,'          `.___,.'  __/_|_  /   / '   |
   _.-_..---""_.-\                            .,...__""--./L/_   |
 -'""'     ,""  ,-`-.    .___.,...___,    _,.+"      """"`-+-==++-
          / /  `.   )"-.._`v \|    V/  /-'    \._,._.'"-. /    /
          ` `.  )---.     `""\\__  / .'        /    \    Y"._.'
           `"'`"     `-.     /|._""_/         |  ,..   _ |  |
                        `"""' |  "'           `. `-'  (_|,-.'
                               \               |`       .`-
                                `.           . j`._    /
                                 |`.._     _.'|    `""/
                                 |    /""'"   |  .". j
                                .`.__j         \ `.' |
                                j    |          `._.'
                               /     |
                              /,  ,  \
                              \|  |   L
                               `..|_..' vh
        ********************************************************
              ********************************************
                    😀  Credit: Hoang Phan Viet  😀
                          ******************
                                ******
    '''
    )
