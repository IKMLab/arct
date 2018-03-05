from ext import pickling  # avoiding circular import again
import torch
import glovar


def load_params(encoder, config):
    transfer_state_dict = encoder_state_dict(config['transfer_name'])
    encoder.load_state_dict(transfer_state_dict)


def encoder_state_dict(transfer_name):
    np_dict = pickling.load(glovar.DATA_DIR, '%s_dict_np' % transfer_name)
    return dict(zip(np_dict.keys(),
                    [torch.from_numpy(a) for a in np_dict.values()]))
