import os
import random

import numpy as np
import pytorch_lightning as pl
import torch

from RANZCR.configure import CFG


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
        CFG.color
        + """
             😀 Cre: Hoang Phan Viet 😀            /
                                 _,.------....___,.' ',.-.
                              ,-'          _,.--'        |
                            ,'         _.-'              .
                           /   ,     ,'                   `
                          .   /     /                     ``.
                          |  |     .                       \.\\
                ____      |___._.  |       __               \ `.
              .'    `---''       ``'-.--''`  \               .  \\
             .  ,            __               `              |   .
             `,'         ,-''  .               \             |    L
            ,'          '    _.'                -._          /    |
           ,`-.    ,'.   `--'                      >.      ,'     |
          . .'\\'   `-'       __    ,  ,-.         /  `.__.-      ,'
          ||:, .           ,'  ;  /  / \ `        `.    .      .'/
          j|:D  \          `--'  ' ,'_  . .         `.__, \   , /
         / L:_  |                 .  '' :_;                `.'.'
         .    '''                  ''''''                    V
          `.                                 .    `.   _,..  `
            `,_   .    .                _,-'/    .. `,'   __  `
             ) \`._        ___....----''  ,'   .'  \ |   '  \  .
            /   `. '`-.--''         _,' ,'     `---' |    `./  |
           .   _  `'''--.._____..--'   ,             '         |
           | .' `. `-.                /-.           /          ,
           | `._.'    `,_            ;  /         ,'          .
          .'          /| `-.        . ,'         ,           ,
          '-.__ __ _,','    '`-..___;-...__   ,.'\ ____.___.'
          `'^--'..'   '-`-^-''--    `-^-'`.'''''''`.,^.`.--' mh
        *********************************************************
    """
    )
