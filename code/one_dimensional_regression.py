import matplotlib.pyplot as plt
import numpy as np
import torch

from net import Net


def test(params, model):

    x = torch.empty(params['test_samples'], params['in_features']).uniform_(params['x_min'], params['x_max'])
    y = model(x)
    y_target = params['f'](x)

    res = {}
    res['x'] = x.detach().numpy()
    res['y'] = y.detach().numpy()
    res['y_target'] = y_target.detach().numpy()

    return res


def train(params, model):

    def batches(params):

        i = 0
        while i < params['train_samples'] - params['batch_size']:
            x = torch.empty(params['batch_size'], params['in_features']).uniform_(params['x_min_train'], params['x_max_train'])
            yield x
            i += params['batch_size']

        x = torch.empty(params['train_samples'] - i, params['in_features']).uniform_(params['x_min_train'], params['x_max_train'])
        yield x

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), params['lr'])

    history_loss = torch.empty(params['train_samples'] // params['batch_size'] + 1)
    history_len_memories = torch.empty(params['train_samples'] // params['batch_size'] + 1)
    for i, x in enumerate(batches(params)):

        y = model(x)
        y_target = params['f'](x)
        loss = criterion(y, y_target)

        model.zero_grad()
        if not model.use_memory_buffer:
            loss.backward()
        else:
            model.memory_buffer.store_memory(x, y_target, loss.item())
            model.memory_buffer.compute_loss_and_backward_pass_for_random_memory(model, criterion)

        optimizer.step()

        history_loss[i] = loss.clone().item()
        history_len_memories[i] = len(model.memory_buffer.memories)

    return {
        'loss': history_loss[:i].detach().numpy(),
        'len_memories': history_len_memories[:i].detach().numpy(),
    }


if __name__ == '__main__':

    params = {
        'seed': 123,
        'in_features': 1,
        'n_hidden': [80, 80],
        'out_features': 1,
        'f': None,
        'test_samples': 500,
        'train_samples': 25000,
        'batch_size': 20,
        'x_min': -10.,
        'x_max': 10.,
        'dx': 2.5,
        'lr': 1e-3,
        'max_buffer_size': 1000,
        'colors': {
            'bp': 'C1',
            'bp full': 'C2',
            'mr': 'C3',
        }
    }

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    # model setup
    model = Net(params['in_features'], params['n_hidden'], params['out_features'])

    model_full = Net(params['in_features'], params['n_hidden'], params['out_features'])
    model_full.clone_parameters(model)

    model_mr = Net(params['in_features'], params['n_hidden'], params['out_features'])
    model_mr.clone_parameters(model)
    model_mr.use_memory_buffer = True
    model_mr.memory_buffer.max_size = params['max_buffer_size'] // params['batch_size']

    def f(x):
        return torch.sin(0.5 * x)

    params['f'] = f

    # initial testing
    res_initial = test(params, model)
    res_initial_full = test(params, model_full)
    res_initial_mr = test(params, model_mr)

    # training
    params['x_min_train'] = params['x_min']
    params['x_max_train'] = params['x_max']
    torch.manual_seed(params['seed'])
    hist_res_full = [train(params, model_full)]

    params['train_samples'] = int(params['train_samples'] // ((params['x_max'] - params['x_min']) / params['dx']))
    hist_res = []
    hist_res_mr = []
    for delta_x_min in np.arange(0., params['x_max'] - params['x_min'] + 1., params['dx']):
        params['x_min_train'] = params['x_min'] + delta_x_min
        params['x_max_train'] = params['x_min'] + delta_x_min + params['dx']
        torch.manual_seed(params['seed'])
        res_train = train(params, model)
        hist_res.append(res_train)
        torch.manual_seed(params['seed'])
        res_train_mr = train(params, model_mr)
        hist_res_mr.append(res_train_mr)

    # final testing
    res_final = test(params, model)
    res_final_full = test(params, model_full)
    res_final_mr = test(params, model_mr)

    # analysis and plotting
    print('bp', np.sum((res_final['y'] - res_final['y_target']) ** 2))
    print('bp full', np.sum((res_final_full['y'] - res_final_full['y_target']) ** 2))
    print('mr', np.sum((res_final_mr['y'] - res_final_mr['y_target']) ** 2))

    fig = plt.figure(figsize=(10, 3))
    ax_loss = fig.add_axes([0.1, 0.15, 0.23, 0.8])
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Training step')
    ax_loss.set_ylabel('Loss')

    ax_buffer_size = fig.add_axes([0.4, 0.15, 0.23, 0.8])
    ax_buffer_size.set_xlabel('Training sample')
    ax_buffer_size.set_ylabel('Memory buffer size')

    ax_values = fig.add_axes([0.7, 0.15, 0.23, 0.8])
    ax_values.set_xlabel(r'$x$')
    ax_values.set_ylabel(r'$f(x)$')

    x_loss_bp = np.arange(len(np.concatenate([r['loss'] for r in hist_res]))) * params['batch_size']
    x_loss_bp_full = np.arange(len(np.concatenate([r['loss'] for r in hist_res_full]))) * params['batch_size']
    ax_loss.plot(x_loss_bp, np.concatenate([r['loss'] for r in hist_res]), label='bp', color=params['colors']['bp'])
    ax_loss.plot(x_loss_bp_full, np.concatenate([r['loss'] for r in hist_res_full]), label='bp full', color=params['colors']['bp full'])
    ax_loss.plot(x_loss_bp, np.concatenate([r['loss'] for r in hist_res_mr]), label='mr', color=params['colors']['mr'])
    ax_loss.legend()

    x_buffer_size = np.arange(len(np.concatenate([r['len_memories'] for r in hist_res_mr]))) * params['batch_size']
    ax_buffer_size.plot(x_buffer_size, np.concatenate([r['len_memories'] for r in hist_res_mr]) * params['batch_size'], label='mr', color=params['colors']['mr'])

    ax_values.plot(res_final['x'], res_final['y_target'], color='k', marker='o', ls='')
    ax_values.plot(res_final['x'], res_final['y'], label='bp', marker='o', ls='', color=params['colors']['bp'])
    ax_values.plot(res_final_full['x'], res_final_full['y'], label='bp full', marker='o', ls='', color=params['colors']['bp full'])
    ax_values.plot(res_final_mr['x'], res_final_mr['y'], label='mr', marker='o', ls='', color=params['colors']['mr'])
    for delta_x_min in np.arange(0., params['x_max'] - params['x_min'] + 1., params['dx']):
        ax_values.axvline(params['x_min'] + delta_x_min, lw=0.5, color='k')
    # ax_values.legend()

    plt.savefig('../figures/one_dimensional_regression.png', dpi=300)
