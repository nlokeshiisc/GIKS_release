import constants
import torch
import torch.nn as nn
from torch.autograd import Function
import utils.common_utils as cu
import torch.nn.functional as F
# from functorch import make_functional


class Truncated_power:
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print("Degree should not set to be 0!")
            raise ValueError

        if not isinstance(self.degree, int):
            print("Degree should be int")
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        try:
            out = torch.zeros(x.shape[0], self.num_of_basis).to(
                cu.get_device(), dtype=torch.float64
            )
        except:
            # This except is required when you run forward with just one example
            out = torch.zeros(1, self.num_of_basis).to(
                cu.get_device(), dtype=torch.float64
            )

        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.0
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = self.relu(x - self.knots[_ - self.degree])
                else:
                    out[:, _] = (
                        self.relu(x - self.knots[_ - self.degree - 1])
                    ) ** self.degree

        return out  # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act="relu", isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis  # num of basis

        self.weight = nn.Parameter(
            torch.rand(self.ind, self.outd, self.d), requires_grad=True
        )

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(
                self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(
            self.weight.T, x_feature.T).T  # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat)  # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2)  # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def comp_grid(y, num_grid):
    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(
            self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out


class Vcnet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots, batch_norm=False):
        super(Vcnet, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(
                    in_features=layer_cfg[0],
                    out_features=layer_cfg[1],
                    bias=layer_cfg[2],
                )
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(
                    nn.Linear(
                        in_features=layer_cfg[0],
                        out_features=layer_cfg[1],
                        bias=layer_cfg[2],
                    )
                )
            density_hidden_dim = layer_cfg[1]

            def apply_act():
                if batch_norm == True:
                    density_blocks.append(nn.BatchNorm1d(density_hidden_dim))

                if layer_cfg[3] == "relu":
                    density_blocks.append(nn.ReLU(inplace=True))
                elif layer_cfg[3] == "tanh":
                    density_blocks.append(nn.Tanh())
                elif layer_cfg[3] == "sigmoid":
                    density_blocks.append(nn.Sigmoid())
                elif layer_cfg[3] == "elu":
                    density_blocks.append(nn.ELU())
                elif layer_cfg[3] == "leaky_relu":
                    density_blocks.append(nn.LeakyReLU(negative_slope=0.1))
                else:
                    print("No activation")

            # For \alpha, take pre activation features only
            if layer_idx != len(cfg_density) - 1:
                apply_act()
            else:
                break

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(
            self.num_grid, density_hidden_dim, isbias=1
        )

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                last_layer = Dynamic_FC(
                    layer_cfg[0],
                    layer_cfg[1],
                    self.degree,
                    self.knots,
                    act=layer_cfg[3],
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                blocks.append(
                    Dynamic_FC(
                        layer_cfg[0],
                        layer_cfg[1],
                        self.degree,
                        self.knots,
                        act=layer_cfg[3],
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )

        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)

    def forward(self, dosage, x, **kwargs):
        embeddings = self.hidden_features(x)

        hidden = nn.ReLU()(embeddings)

        t_hidden = torch.cat((torch.unsqueeze(dosage, 1), hidden), 1)
        # t_hidden = torch.cat((torch.unsqueeze(t, 1), x), 1)

        g = self.density_estimator_head(dosage, hidden)

        Q = self.Q(t_hidden)

        if (
            constants.RETURN_EMB in kwargs.keys()
            and kwargs[constants.RETURN_EMB] == True
        ):
            return g, Q, embeddings
        return g, Q

    def forward_with_emb(self, dosage, x_emb, **kwargs):
        t_hidden = torch.cat((torch.unsqueeze(dosage, 1), x_emb), 1)
        g = self.density_estimator_head(dosage, x_emb)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.0)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()


class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis  # num of basis
        self.weight = nn.Parameter(torch.rand(self.d), requires_grad=True)

    def forward(self, t, **kwargs):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        # self.weight.data.normal_(0, 0.01)
        self.weight.data.zero_()


# ------------------------------------------ Drnet and Tarnet ------------------------------------------- #


class Treat_Linear(nn.Module):
    def __init__(self, ind, outd, act="relu", istreat=1, isbias=1, islastlayer=0):
        super(Treat_Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias
        self.istreat = istreat
        self.islastlayer = islastlayer

        self.weight = nn.Parameter(torch.rand(
            self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if self.istreat:
            self.treat_weight = nn.Parameter(
                torch.rand(1, self.outd), requires_grad=True
            )
        else:
            self.treat_weight = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x, **kwargs):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, [0]]

        out = torch.matmul(x_feature, self.weight)

        if self.istreat:
            out = out + torch.matmul(x_treat, self.treat_weight)
        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        if not self.islastlayer:
            out = torch.cat((x_treat, out), 1)

        return out


class Multi_head(nn.Module):
    def __init__(self, cfg, isenhance):
        super(Multi_head, self).__init__()

        self.cfg = cfg  # cfg does NOT include the extra dimension of concat treatment
        # set 1 to concat treatment every layer/ 0: only concat on first layer
        self.isenhance = isenhance

        # we default set num of heads = 5
        self.pt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        self.outdim = -1
        # construct the predicting networks
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                self.outdim = layer_cfg[1]
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q1 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q2 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q3 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q4 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(
                    layer_cfg[0],
                    layer_cfg[1],
                    act=layer_cfg[3],
                    istreat=istreat,
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(
                        layer_cfg[0],
                        layer_cfg[1],
                        act=layer_cfg[3],
                        istreat=istreat,
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)
        self.Q5 = nn.Sequential(*blocks)

    def forward(self, x, **kwargs):
        # x = [treatment, features]
        out = torch.zeros(x.shape[0], self.outdim).to(
            cu.get_device(), dtype=torch.float64
        )
        t = x[:, 0]

        def find_idx(arr, gt, lt): return list(
            set(torch.where(arr >= gt)[0].tolist())
            & set(torch.where(arr < lt)[0].tolist())
        )

        idx1 = find_idx(t, self.pt[0], self.pt[1])
        idx2 = find_idx(t, self.pt[1], self.pt[2])
        idx3 = find_idx(t, self.pt[2], self.pt[3])
        idx4 = find_idx(t, self.pt[3], self.pt[4])
        idx5 = find_idx(t, self.pt[4], self.pt[5])

        if idx1:
            out1 = self.Q1(x[idx1, :])
            out[idx1, :] = out[idx1, :] + out1

        if idx2:
            out2 = self.Q2(x[idx2, :])
            out[idx2, :] = out[idx2, :] + out2

        if idx3:
            out3 = self.Q3(x[idx3, :])
            out[idx3, :] = out[idx3, :] + out3

        if idx4:
            out4 = self.Q4(x[idx4, :])
            out[idx4, :] = out[idx4, :] + out4

        if idx5:
            out5 = self.Q5(x[idx5, :])
            out[idx5, :] = out[idx5, :] + out5

        return out


class Drnet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance):
        super(Drnet, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(
                    in_features=layer_cfg[0],
                    out_features=layer_cfg[1],
                    bias=layer_cfg[2],
                )
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(
                    nn.Linear(
                        in_features=layer_cfg[0],
                        out_features=layer_cfg[1],
                        bias=layer_cfg[2],
                    )
                )
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == "relu":
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == "tanh":
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == "sigmoid":
                density_blocks.append(nn.Sigmoid())
            else:
                print("No activation")

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(
            self.num_grid, density_hidden_dim, isbias=1
        )

        # multi-head outputs blocks
        self.Q = Multi_head(cfg, isenhance)

    def forward(self, dosage, x, **kwargs):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(dosage, 1), hidden), 1)
        g = self.density_estimator_head(dosage, hidden)
        Q = self.Q(t_hidden)

        if (
            constants.RETURN_EMB in kwargs.keys()
            and kwargs[constants.RETURN_EMB] == True
        ):
            return g, Q, hidden
        return g, Q

    def forward_with_emb(self, dosage, x_emb, **kwargs):
        t_hidden = torch.cat((torch.unsqueeze(dosage, 1), x_emb), 1)
        g = self.density_estimator_head(dosage, x_emb)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(
                        0, 1.0
                    )  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()


class SqueezeLayer(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return


class ReverseLayerF(Function):
    """This is used to do gradient reversal and thus perform adversarial training
    Code credits: https://github.com/fungtion/DANN
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class FNN(nn.Module):
    """creates a Feed Forward Neural network with the specified Architecture
    nn ([type]): [description]
    """

    def __init__(self, in_dim, out_dim, nn_arch, prefix, *args, **kwargs):
        """Creates a basic embedding block
        Args:
            in_dim ([type]): [description]
            embed_arch ([type]): [description]
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nn_arch = nn_arch
        self.prefix = prefix

        self.model = nn.Sequential()

        prev = in_dim
        for idx, hdim in enumerate(nn_arch):
            self.model.add_module(
                f"{self.prefix}-emb_hid_{idx}", nn.Linear(prev, hdim))
            self.model.add_module(
                f"{self.prefix}-lReLU_{idx}", nn.LeakyReLU(inplace=True)
            )
            prev = hdim
        self.model.add_module(
            f"{self.prefix}-last_layer", nn.Linear(prev, out_dim))

    def forward(self, x):
        return self.model(x)


class DANN(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance):
        super(DANN, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(
                    in_features=layer_cfg[0],
                    out_features=layer_cfg[1],
                    bias=layer_cfg[2],
                )
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(
                    nn.Linear(
                        in_features=layer_cfg[0],
                        out_features=layer_cfg[1],
                        bias=layer_cfg[2],
                    )
                )
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == "relu":
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == "tanh":
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == "sigmoid":
                density_blocks.append(nn.Sigmoid())
            else:
                print("No activation")

        self.hidden_features = nn.Sequential(*density_blocks)

        self.domain_cls = FNN(
            in_dim=density_hidden_dim, out_dim=1, nn_arch=[50], prefix="dann"
        )

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(
            self.num_grid, density_hidden_dim, isbias=1
        )

        # multi-head outputs blocks
        self.Q = Multi_head(cfg, isenhance)

    def forward(self, dosage, x, **kwargs):
        try:
            alpha = kwargs[constants.DANN_ALPHA]
        except:
            alpha = 0

        hidden = self.hidden_features(x)
        hidden_reverse = ReverseLayerF.apply(hidden, alpha)
        domain_pred = torch.squeeze(self.domain_cls(hidden_reverse))
        t_hidden = torch.cat((torch.unsqueeze(dosage, 1), hidden), 1)
        g = self.density_estimator_head(dosage, hidden)
        Q = self.Q(t_hidden)

        if (
            constants.RETURN_EMB in kwargs.keys()
            and kwargs[constants.RETURN_EMB] == True
        ):
            return g, Q, hidden, domain_pred
        return g, Q, domain_pred

    def forward_with_emb(self, t, x_emb, **kwargs):
        try:
            alpha = kwargs[constants.DANN_ALPHA]
        except:
            alpha = 0

        hidden_reverse = ReverseLayerF.apply(x_emb, alpha)
        domain_pred = torch.squeeze(self.domain_cls(hidden_reverse))
        t_hidden = torch.cat((torch.unsqueeze(t, 1), x_emb), 1)
        g = self.density_estimator_head(t, x_emb)
        Q = self.Q(t_hidden)
        return g, Q, domain_pred

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(
                        0, 1.0
                    )  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()


class TreatEmbedding(nn.Module):
    def __init__(self, d_model: int, beta_dim):
        super().__init__()
        self.beta_dims = beta_dim
        self.Emb = nn.Embedding(beta_dim, d_model)

    def forward(self, beta):
        return torch.squeeze(self.Emb(beta))


class ContTreatEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.Emb = nn.Linear(1, d_model)

    def forward(self, beta):
        return torch.squeeze(self.Emb(beta))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class IRM(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance):
        super(IRM, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(
                    in_features=layer_cfg[0],
                    out_features=layer_cfg[1],
                    bias=layer_cfg[2],
                )
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(
                    nn.Linear(
                        in_features=layer_cfg[0],
                        out_features=layer_cfg[1],
                        bias=layer_cfg[2],
                    )
                )
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == "relu":
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == "tanh":
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == "sigmoid":
                density_blocks.append(nn.Sigmoid())
            else:
                print("No activation")

        self.hidden_features = nn.Sequential(*density_blocks)

        self.treat_embedding = ContTreatEmbedding(d_model=50)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(
            self.num_grid, density_hidden_dim, isbias=1
        )

        # multi-head outputs blocks
        self.Q = Multi_head(cfg, isenhance)

    def forward(self, dosage, x, **kwargs):
        try:
            irm_w = kwargs[constants.IRM_W]
        except:
            irm_w = torch.tensor(
                1.0
            ).cuda()  # If you do not pass, I will not flow the gradients

        hidden = self.hidden_features(x)
        hidden = hidden * irm_w

        t_hidden = torch.cat((torch.unsqueeze(dosage, 1), hidden), 1)
        g = self.density_estimator_head(dosage, hidden)
        Q = self.Q(t_hidden)

        if (
            constants.RETURN_EMB in kwargs.keys()
            and kwargs[constants.RETURN_EMB] == True
        ):
            return g, Q, hidden
        return g, Q

    def forward_with_emb(self, t, x_emb, **kwargs):
        try:
            irm_w = kwargs[constants.IRM_W]
        except:
            irm_w = torch.tensor(
                1.0
            ).cuda()  # If you do not pass, I will not flow the gradients

        x_emb = x_emb * irm_w

        t_hidden = torch.cat((torch.unsqueeze(t, 1), x_emb), 1)
        g = self.density_estimator_head(t, x_emb)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(
                        0, 1.0
                    )  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()


class GP_NN:
    def __init__(self):
        """GP has no parameters as of now"""
        self.ml_primal = {}  # This contains the loss w.r.t. the datasets
        self.ker_inv = {}
        self.mean = {}

    def forward(self, Z_f: torch.Tensor, **kwargs):
        with torch.no_grad():
            beta = torch.ones(1, 1).to(cu.get_device(), dtype=torch.float64)
            lam = 1 * torch.ones(1, 1).to(cu.get_device(), dtype=torch.float64)
            r = (
                beta / lam
            )  # This is some kind of normalization on the embeddings obtained.

            self.DD = Z_f.shape[1]

            if kwargs[constants.GP_KERNEL] == constants.DOTPRODUCT_KERNEL:
                phi_phi = torch.matmul(Z_f.T, Z_f)
            elif kwargs[constants.GP_KERNEL] == constants.COSINE_KERNEL:
                Z_f_norm = F.normalize(Z_f, p=2, dim=1)
                phi_phi = torch.matmul(Z_f_norm.T, Z_f_norm)

            Ker = r * phi_phi + torch.eye(Z_f.shape[1], dtype=torch.float64).to(
                cu.get_device(), dtype=torch.float64
            )

            # A very weird pytorch bug: https://github.com/pytorch/pytorch/issues/70669
            try:
                L_matrix = torch.linalg.cholesky(Ker)
            except:
                L_matrix = torch.linalg.cholesky(Ker)

            # Lokesh: Check this :: This is correct
            L_inv = torch.linalg.solve_triangular(
                L_matrix,
                torch.eye(self.DD, dtype=torch.float64).to(
                    cu.get_device(), dtype=torch.float64
                ),
                upper=False,
            )

            # L_y = torch.matmul(L_inv_reduce, torch.matmul(Z_f.T, Y_f))
            L_phi = torch.matmul(L_inv, Z_f.T)

            self.ker_inv = torch.matmul(L_inv.T, L_inv) / lam

            # self.mean[key] = r * torch.matmul(L_inv_reduce.T, L_y)
            self.Linv_Lphi = r * torch.matmul(L_inv.T, L_phi)

    def mean_w(self, Y_f):
        """Computes the mean i.e. w"""
        return torch.matmul(self.Linv_Lphi, Y_f.T)

    def element_variance(self, Z_f, index, ctr_prefix: str = "fct-"):
        key = f"{ctr_prefix}{index}"
        return torch.diag(torch.matmul(Z_f, torch.matmul(self.ker_inv[key], Z_f.T)), 0)


class SDotAttn(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.

    Thanks to: https://github.com/sooftware/attentions/blob/master/attentions.py
    """

    def __init__(self, input_dim: int, attn_dim: int):
        super(SDotAttn, self).__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(attn_dim))
        # Create the projecyion matrices
        self.WQ = nn.Linear(in_features=input_dim,
                            out_features=attn_dim, bias=False)
        self.WK = nn.Linear(in_features=input_dim,
                            out_features=attn_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Computes the scaled dot-product attention and returns the attention scores and the aggregated ycf.
        The attention scores are normalized.

        Args:
            query (torch.Tensor): _description_
            key (torch.Tensor): _description_
            value (torch.Tensor): _description_

        Returns:
            normalized attention scores
            aggregated y
        """
        assert len(
            query.shape) == 1, "We assume only one query at a time in attention"
        qProj = self.WQ(query)
        kProj = self.WK(key)

        score = qProj.view(1, -1) @ kProj.T / self.sqrt_dim

        attn = F.softmax(score, -1)
        aggr_ycf = (attn @ value).squeeze()
        return attn, aggr_ycf


class GP_JOINT_X_T(nn.Module):
    def __init__(self, kernel=constants.COSINE_KERNEL):
        """This GP implements a product kernel.
        K( (x1, t1), (x2, t2) ) = K(x1, x2) * K(t1, t2)
        On x, we have the usual cosine kernel on embeddings of features.
        On t, K(t1, t2) = Relu(\alpha - |t1-t2|)
        \alpha is a learnabe parameter that is tuned on the validation dataset."""
        super(GP_JOINT_X_T, self).__init__()
        self.kernel = kernel
        self.alpha = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.eta = nn.Parameter(torch.Tensor([10]), requires_grad=True)

    def forward(self, Z_f: torch.Tensor, t: torch.Tensor, **kwargs):
        beta = torch.ones(1, 1).to(cu.get_device(), dtype=torch.float64)
        lam = 1 * torch.ones(1, 1).to(cu.get_device(), dtype=torch.float64)
        self.r = (
            beta / lam
        )  # This is some kind of normalization on the embeddings obtained.

        if self.kernel == constants.COSINE_KERNEL:
            Z_f_norm = F.normalize(Z_f, p=2, dim=1)
            phi_phi = torch.matmul(Z_f_norm, Z_f_norm.T)
        elif self.kernel == constants.RBF_KERNEL:
            # Re-indexing
            X_i = Z_f[:, None, :]  # shape (N, D) -> (N, 1, D)
            Y_j = Z_f[None, :, :]  # shape (N, D) -> (1, N, D)

            sqd = torch.sum((X_i - Y_j) ** 2, 2)  # |X_i - Y_j|^2
            phi_phi = torch.exp(-0.5 * sqd)  # Gaussian Kernel

            assert phi_phi.shape == (len(Z_f), len(Z_f))

        ker_t = torch.relu(
            self.alpha - self.eta * torch.abs(t.view(-1, 1) - t.view(1, -1))
        )
        assert ker_t.shape == phi_phi.shape

        # multiply phi_phi and ker_t element wise
        self.phi = phi_phi * ker_t
        self.NN = len(t)

        self.sig_p_NN = self.r * torch.eye(
            self.NN, dtype=torch.float64, device=cu.get_device()
        )

        # Ker = self.phi.T @ self.sig_p_DD @ self.phi    #lamda= 1/sig^2 = 1/r
        Ker = self.r * self.phi.T @ self.phi
        Ker = Ker + self.sig_p_NN

        # A very weird pytorch bug: https://github.com/pytorch/pytorch/issues/70669
        try:
            L_matrix = torch.linalg.cholesky(Ker)
        except:
            L_matrix = torch.linalg.cholesky(Ker)

        L_inv = torch.linalg.solve_triangular(
            L_matrix,
            torch.eye(self.NN, dtype=torch.float64).to(
                cu.get_device(), dtype=torch.float64
            ),
            upper=False,
        )

        self.ker_inv = torch.matmul(L_inv.T, L_inv)  # / lam
        self.sig_phi_Kinv = self.r * self.phi @ self.ker_inv

    def mean_var(self, Y_f, Z_f, t, Z_f_star, t_star, **kwargs):
        """Computes the mean, and variance together"""
        if len(Z_f_star.shape) == 1:
            Z_f_star = Z_f_star.view(1, -1)
        if len(t_star.shape) == 0:
            t_star = t_star.view(-1)

        phi_phi = torch.matmul(Z_f, Z_f_star.T)

        ker_t_star = torch.relu(
            self.alpha - torch.abs(t.view(-1, 1) - t_star.view(1, -1))
        )
        phi_star = phi_phi * ker_t_star

        mean = phi_star.T @ self.sig_phi_Kinv @ Y_f
        var = (1 / self.r) * (self.r * phi_star.T @ phi_star) - (
            self.r
            * self.r
            * phi_star.T
            @ self.phi
            @ self.ker_inv
            @ self.phi.T
            @ phi_star
        )

        return mean, torch.diag(var)


"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license
Python Implementation of the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
We provide both biased estimator and unbiased estimators (unbiased estimator is used in the paper)
"""


def to_numpy(x):
    """convert Pytorch tensor to numpy array"""
    return x.clone().detach().cpu().numpy()


class HSIC(nn.Module):
    """Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.
    Empirically, we use the finite sample estimator of HSIC (with m observations) by,
    (1) biased estimator (HSIC_0)
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        :math: (m - 1)^2 tr KHLH.
        where K_{ij} = kernel_x (x_i, x_j), L_{ij} = kernel_y (y_i, y_j), H = 1 - m^{-1} 1 1 (Hence, K, L, H are m by m matrices).
    (2) unbiased estimator (HSIC_1)
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        :math: \frac{1}{m (m - 3)} \bigg[ tr (\tilde K \tilde L) + \frac{1^\top \tilde K 1 1^\top \tilde L 1}{(m-1)(m-2)} - \frac{2}{m-2} 1^\top \tilde K \tilde L 1 \bigg].
        where \tilde K and \tilde L are related to K and L by the diagonal entries of \tilde K_{ij} and \tilde L_{ij} are set to zero.
    Parameters
    ----------
    sigma_x : float
        the kernel size of the kernel function for X.
    sigma_y : float
        the kernel size of the kernel function for Y.
    algorithm: str ('unbiased' / 'biased')
        the algorithm for the finite sample estimator. 'unbiased' is used for our paper.
    reduction: not used (for compatibility with other losses).
    """

    def __init__(self, sigma_x, sigma_y=None, algorithm="unbiased", reduction=None):
        super(HSIC, self).__init__()

        if sigma_y is None:
            sigma_y = sigma_x

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        if algorithm == "biased":
            self.estimator = self.biased_estimator
        elif algorithm == "unbiased":
            self.estimator = self.unbiased_estimator
        else:
            raise ValueError("invalid estimator: {}".format(algorithm))

    def _kernel_x(self, X):
        raise NotImplementedError

    def _kernel_y(self, Y):
        raise NotImplementedError

    def biased_estimator(self, input1, input2):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        K = self._kernel_x(input1)
        L = self._kernel_y(input2)

        KH = K - K.mean(0, keepdim=True)
        LH = L - L.mean(0, keepdim=True)

        N = len(input1)

        return torch.trace(KH @ LH / (N - 1) ** 2)

    def unbiased_estimator(self, input1, input2):
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        kernel_XX = self._kernel_x(input1)
        kernel_YY = self._kernel_y(input2)

        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)

        N = len(input1)

        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )

        return hsic / (N * (N - 3))

    def forward(self, input1, input2, **kwargs):
        return self.estimator(input1, input2)


class RbfHSIC(HSIC):
    """Radial Basis Function (RBF) kernel HSIC implementation."""

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma**2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    def _kernel_x(self, X):
        return self._kernel(X, self.sigma_x)

    def _kernel_y(self, Y):
        return self._kernel(Y, self.sigma_y)
