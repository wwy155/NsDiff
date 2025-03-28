
import torch
import torch.nn as nn
import src.nn.csbi_sde as sde
import src.nn.csbi_util as util
from src.nn.csbi_net import Transformerv5, build_transformerv5


def build(opt, dyn, direction):
    print(util.magenta("build {} policy...".format(direction)))

    net_name = getattr(opt, direction+'_net')
    zero_out_last_layer = (direction=='forward')
    net = _build_net(opt, net_name=net_name, zero_out_last_layer=zero_out_last_layer)
    use_t_idx = (net_name in ['toy', 'Unet', 'Unetv2', 'Unetv2_large', 'DGLSB',
        'Transformerv1', 'Transformerv2', 'Transformerv3', 'Transformerv4', 'Transformerv5'])
    # t_idx is handled internally in ncsnpp
    # scale_by_g = (net_name in ['ncsnpp'])
    policy = SchrodingerBridgePolicy(
        opt, direction, dyn, net, use_t_idx=use_t_idx, scale_by_g=opt.scale_by_g)

    print(util.red(f'number of parameters is {util.count_parameters(policy)/1000:.3f}k'))
    policy.to(opt.device)

    return policy

class ZeroNet(nn.Module):
    """To keep consistent with ema, optimizer, or sheduler framework, we add a fake parameter,
    which will never be used.
    """
    # `zero_out_last_layer` is useless but consistent with the `policy` format.
    def __init__(self, zero_out_last_layer=True):
        super().__init__()
        self.fake_net = nn.Parameter(torch.zeros(1))
    @ property
    def zero_out_last_layer(self):
        return True
    def forward(self, x, t):
        return torch.zeros_like(x)

def _build_net(opt, zero_out_last_layer=True, net_name='Transformerv5'):
    compute_sigma = lambda t: sde.compute_sigmas(t, opt.sigma_min, opt.sigma_max)
    # zero_out_last_layer = opt.DSM_warmup
    # if net_name not in [None, 'Zero', 'toy']:
    #     print(f'Model config: [{net_name}]\n', opt.model_configs[net_name])

    if net_name in [None, 'Zero']:
        net = ZeroNet()
    # elif net_name == 'toy':
    #     assert util.is_toy_dataset(opt)
    #     from models.toy_model.Toy import build_toy
    #     net = build_toy(zero_out_last_layer)
    # elif net_name == 'toyv2':
    #     from models.toy_model.Toyv2 import build_toyv2
    #     net = build_toyv2(opt.model_configs[net_name])
    # elif net_name == 'toyv3':
    #     from models.toy_model.Toyv3 import build_toyv3
    #     net = build_toyv3(opt.model_configs[net_name])
    # elif net_name == 'Unet':
    #     from models.Unet.Unet import build_unet
    #     net = build_unet(opt.model_configs[net_name], zero_out_last_layer)
    # elif net_name in ['Unetv2', 'Unetv2_large']:
    #     from models.Unet.Unetv2 import build_unetv2
    #     net = build_unetv2(opt.model_configs[net_name], zero_out_last_layer)
    # elif net_name == 'ncsnpp':
    #     from models.ncsnpp.ncsnpp import build_ncsnpp
    #     net = build_ncsnpp(opt.model_configs[net_name], compute_sigma, zero_out_last_layer)
    # elif net_name == 'DGLSB':
    #     from models.DGLSB.dglsb import build_dglsb
    #     net = build_dglsb(zero_out_last_layer)
    # elif net_name == 'Transformerv0':
    #     from models.Transformer.Transformerv0 import build_transformerv0
    #     net = build_transformerv0(opt.model_configs[net_name], opt.interval, zero_out_last_layer,
    #         fix_embedding=False)
    # elif net_name == 'Transformerv1':
    #     from models.Transformer.Transformerv1 import build_transformerv1
    #     net = build_transformerv1(opt.model_configs[net_name], opt.interval, zero_out_last_layer,
    #         fix_embedding=False)
    # elif net_name == 'Transformerv2':
    #     from models.Transformer.Transformerv2 import build_transformerv2
    #     net = build_transformerv2(opt.model_configs[net_name], opt.interval, zero_out_last_layer)
    # elif net_name == 'Transformerv3':
    #     from models.Transformer.Transformerv3 import build_transformerv3
    #     net = build_transformerv3(opt.model_configs[net_name], opt.interval, zero_out_last_layer)
    # elif net_name == 'Transformerv4':
    #     from models.Transformer.Transformerv4 import build_transformerv4
    #     net = build_transformerv4(opt.model_configs[net_name], opt.interval, zero_out_last_layer)
    elif net_name == 'Transformerv5':
        getattr(opt.model_configs, net_name).input_size = opt.input_size
        net = build_transformerv5(getattr(opt.model_configs, net_name), opt.interval, zero_out_last_layer)
    else:
        raise NotImplementedError(f'New model {net_name}')
    return net

class SchrodingerBridgePolicy(torch.nn.Module):
    # note: scale_by_g matters only for pre-trained model
    def __init__(self, opt, direction, dyn, net, use_t_idx=False, scale_by_g=True):
        super(SchrodingerBridgePolicy,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx
        self.scale_by_g = scale_by_g

    @ property
    def zero_out_last_layer(self):
        return self.net.zero_out_last_layer


    def forward(self, x, t):
        # make sure t.shape = [batch], t always continuous value in [0, T]
        t = t.squeeze()
        if getattr(self.opt, self.direction + '_net') in ['Transformerv2', 'Transformerv3',
            'Transformerv4', 'Transformerv5']:
            num_batches = x[0].shape[0]  # x --> (obs, cond_mas)
            x_dim = x[0].dim()
        else:
            num_batches = x.shape[0]
            x_dim = x.dim()
        if t.dim()==0:
            t = t.repeat(num_batches)
        assert t.dim() == 1
        assert t.shape[0] == num_batches

        if self.use_t_idx:
            # t_input = t / self.opt.T * self.opt.interval  # t_input as indices.
            # ts was created in Runner self.ts = torch.linspace(opt.t0, opt.T, opt.interval)
            # The index will not be converted back to the original index.
            # Examples: (some numbers are removed)
            # tensor([0.0010, 0.0214, 0.0418, 0.0622, 0.0826, 0.1029, 0.1233, 0.1437, 0.1641,
            #         0.7350, 0.7553, 0.7757, 0.7961, 0.8165, 0.8369, 0.8573, 0.8777, 0.8981,
            #         0.9184, 0.9388, 0.9592, 0.9796, 1.0000])
            # tensor([ 0.0500,  1.0694,  2.0888,  3.1082,  4.1276,  5.1469,  6.1663,  7.1857,
            #         40.8255, 41.8449, 42.8643, 43.8837, 44.9031, 45.9225, 46.9418, 47.9612,
            #         48.9806, 50.0000])
            dt = (self.opt.T - self.opt.t0) / (self.opt.interval-1)
            t_input = (t-self.opt.t0) / dt

        else:
            t_input = t  # t_input as continous.

        out = self.net(x, t_input)

        # if the SB policy behaves as "Z" in FBSDE system,
        # the output should be scaled by the diffusion coefficient "g".
        if self.scale_by_g:
            g = self.dyn.g(t)
            g = g.reshape(num_batches, *([1,]*(x_dim-1)))
            out = out * g
            # print('scale g', g.reshape(-1))

        return out


