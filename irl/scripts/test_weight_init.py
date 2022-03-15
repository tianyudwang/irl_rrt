import torch as th
import torch.nn as nn  

import irl.utils.pytorch_utils as ptu

def init_weights(m):
    if isinstance(m, nn.Linear):
        th.nn.init.normal_(m.weight, mean=100.0)
        # th.nn.init.constant_(m.weight, 100.0)
        m.bias.data.fill_(100.0)

def main():
    model = ptu.build_mlp(
        input_size=6,
        output_size=1,
        n_layers=2,
        size=32,
        activation='relu',
        output_activation='identity'
    ).to('cuda')

    model.apply(init_weights)

    count = 0
    for i in range(1000):
        x = (th.rand((1, 6), device='cuda') - 0.5) * 100

        out = model(x).item()
        if out > 0:
            count += 1 

    print(count)



if __name__ == '__main__':
    main()