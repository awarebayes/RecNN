def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


def write_losses(writer, loss_dict, kind='train'):

    def write_loss(kind, key, item, step):
        writer.add_scalar(kind + '/' + key, item, global_step=step)

    step = loss_dict['step']
    for k, v in loss_dict.items():
        if k == 'step':
            continue
        write_loss(kind, k, v, step)

    writer.close()


class DummyWriter:
    def add_figure(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
