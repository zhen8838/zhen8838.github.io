
# '| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'


from tqdm import trange, tqdm
from random import random, randint
from time import sleep


def training(epoch: int, step_per_epoch: int):
    for i in range(epoch):
        with tqdm(total=step_per_epoch, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt} {postfix}', ncols=80) as t:
            for i in range(step_per_epoch):
                t.set_postfix_str('loss={:^7.3f}'.format(random()))
                sleep(.1)
                t.update()


training(epoch=3, step_per_epoch=10)
# from tqdm import trange, tqdm
# from random import random, randint
# from time import sleep

# def training(epoch: int, step_per_epoch: int):
#     for i in range(epoch):
#         with tqdm(total=step_per_epoch, bar_format='{l_bar}{bar}| {rate_fmt} {postfix[0]}{postfix[1][loss]:>6.3f}',
#                   unit=' batch', postfix=['loss=', dict(loss=0)], ncols=80) as t:
#             for j in range(step_per_epoch):
#                 t.postfix[1]["loss"] = random()
#                 t.update()
# training(epoch=3, step_per_epoch=10)


# from tqdm import tqdm
# from random import random, randint
# from time import sleep


# def training(epoch: int, step_per_epoch: int):
#     for i in range(epoch):
#         with tqdm(total=step_per_epoch, ncols=80) as t:
#             for j in range(step_per_epoch):
#                 t.set_postfix(loss='{:^7.3f}'.format(random()))
#                 sleep(0.1)
#                 t.update()


# training(epoch=3, step_per_epoch=10)
