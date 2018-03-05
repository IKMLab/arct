"""Search spaces for grid search."""


def space(config):
    if config['name'] == 'trans100frozengrid':
        return {
            'lr': [0.2, 0.1, 0.09, 0.06, 0.03],
            'l2': [0.1, 0.01, 0.001, 0.],
            'p_drop': [0.1, 0.01, 0.001, 0.],
            'tune_embeds': [False],
            'tune_encoder': [False],
            'transfer': [True],
            'encoder_size': [100],
            'transfer_name': ['100grid1'],
            'hidden_size': [100, 300, 512]
        }
    elif config['name'] == 't2048f_grid':
        return {
            'lr': [0.2, 0.1, 0.09, 0.06, 0.03],
            'l2': [0.1, 0.01, 0.001, 0.],
            'p_drop': [0.1, 0.01, 0.001, 0.],
            'tune_embeds': [False],
            'tune_encoder': [False],
            'transfer': [True],
            'encoder_size': [2048],
            'transfer_name': ['bilstm2048'],
            'hidden_size': [100, 300, 512]
        }
    # Linear 2048 Transferred Tuned
    elif config['name'] == 't2048lin_grid':
        return {
            'lr': [0.004, 0.003, 0.002, 0.001],
            'l2': [0.],
            'p_drop': [0.1, 0.01],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [2048],
            'transfer_name': ['bilstm2048'],
            'transfer': [True]
        }
    # Linear 2048 Random Tuned
    elif config['name'] == 'r2048lin_grid':
        return {
            'lr': [0.004, 0.003, 0.002, 0.001],
            'l2': [0.],
            'p_drop': [0.1, 0.01],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [2048],
            'transfer': [False]
        }
    # Linear 100 Transferred Tuned
    elif config['name'] == 't100lin_grid':
        return {
            'lr': [0.2, 0.1, 0.09, 0.06, 0.03, 0.01],
            'l2': [0.],
            'p_drop': [0.1, 0.01, 0.],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1, 0.01],
            'tune_encoder': [True],
            'enc_lr_factor': [0.1, 0.01],
            'encoder_size': [100],
            'transfer_name': ['100grid1'],
            'transfer': [True]
        }
    elif config['name'] == 'r100lin_grid':
        return {
            'lr': [0.2, 0.1, 0.09, 0.06, 0.03, 0.01],
            'l2': [0.],
            'p_drop': [0.1, 0.01, 0.],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1, 0.01],
            'tune_encoder': [True],
            'enc_lr_factor': [0.1, 0.01],
            'encoder_size': [100],
            'transfer': [False]
        }
    # Frozens for the above...

    #
    elif config['name'] == 't100compc_grid':
        return {
            'lr': [0.2, 0.1, 0.09, 0.06, 0.03],
            'l2': [0.1, 0.01, 0.],
            'p_drop': [0.1, 0.01, 0.],
            'tune_embeds': [True],
            'tune_encoder': [True],
            'transfer': [True],
            'encoder_size': [100],
            'transfer_name': ['100grid1'],
            'hidden_size': [512]
        }
    #
    elif config['name'] == 't2048compc_grid':
        return {
            'lr': [0.003, 0.002, 0.001],
            'l2': [0.01, 0.],
            'p_drop': [0.1, 0.],
            'tune_embeds': [True],
            'tune_encoder': [True],
            'transfer': [True],
            'encoder_size': [2048],
            'transfer_name': ['bilstm2048'],
            'hidden_size': [512]
        }
    #
    elif config['name'] == 't300comp_grid':
        return {
            'lr': [0.007, 0.006, 0.005, 0.004],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1, 0.01],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [300],
            'transfer_name': ['e300'],
            'transfer': [True],
            'hidden_size': [512]
        }
    elif config['name'] == 't300lin_grid':
        return {
            'lr': [0.03, 0.02, 0.01],  # originally as high as 0.09
            'p_drop': [0.1, 0.01, 0.],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1],
            'tune_encoder': [True],
            'enc_lr_factor': [1., 0.1],
            'encoder_size': [300],
            'transfer_name': ['e300'],
            'transfer': [True]
        }
    elif config['name'] == 't512comp_grid':
        return {
            'lr': [0.004, 0.003, 0.002, 0.001],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1, 0.01],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True],
            'hidden_size': [512]
        }
    elif config['name'] == 't512lin_grid':
        return {
            'lr': [0.04, 0.03, 0.02, 0.01],
            'p_drop': [0.1],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1],
            'tune_encoder': [True],
            'enc_lr_factor': [1., 0.1],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True]
        }
    elif config['name'] == 't1024comp_grid':
        return {
            'lr': [0.004, 0.003, 0.002, 0.001],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1, 0.01],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [1024],
            'transfer_name': ['e1024'],
            'transfer': [True],
            'hidden_size': [512]
        }
    elif config['name'] == 't1024lin_grid':
        return {
            'lr': [0.04, 0.03, 0.02, 0.01],
            'p_drop': [0.1],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1],
            'tune_encoder': [True],
            'enc_lr_factor': [1., 0.1],
            'encoder_size': [1024],
            'transfer_name': ['e1024'],
            'transfer': [True]
        }
    elif config['name'] == 't512compc_grid':
        return {
            'lr': [0.005, 0.004, 0.003, 0.002],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [True],
            'emb_lr_factor': [0.1],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True],
            'hidden_size': [512]
        }

    #
    # FROZEN EMBEDDINGS

    #
    # Comp Model

    # 2048

    elif config['name'] == 't2048fwcomp_grid':
        return {
            'lr': [0.004, 0.003, 0.002, 0.001, 0.0009],  # 0.002
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [2048],
            'transfer_name': ['e2048'],
            'transfer': [True],
            'hidden_size': [512]
        }

    # 1024

    elif config['name'] == 't1024fwcomp_grid':
        return {
            'lr': [0.005, 0.004, 0.003, 0.002, 0.001],  # 0.004
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [1024],
            'transfer_name': ['e1024'],
            'transfer': [True],
            'hidden_size': [512]
        }

    # 512

    elif config['name'] == 't512fwcomp_grid':
        return {
            'lr': [0.006, 0.005, 0.004, 0.003, 0.002],  # 0.003
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True],
            'hidden_size': [512]
        }

    # 300

    elif config['name'] == 't300fwcomp_grid':
        return {
            'lr': [0.006, 0.005, 0.004, 0.003, 0.002],  # 0.002
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [300],
            'transfer_name': ['e300'],
            'transfer': [True],
            'hidden_size': [512]
        }

    # 100

    elif config['name'] == 't100fwcomp_grid':
        return {
            'lr': [0.006, 0.005, 0.004, 0.003, 0.002],  # 0.003
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [100],
            'transfer_name': ['e100'],
            'transfer': [True],
            'hidden_size': [512]
        }

    #
    # Lin Model

    # 2048

    elif config['name'] == 't2048fwlin_grid':
        return {
            'lr': [0.005, 0.004, 0.003, 0.002, 0.001],  # 0.002
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [2048],
            'transfer_name': ['e2048'],
            'transfer': [True]
        }

    # 1024

    elif config['name'] == 't1024fwlin_grid':
        return {
            'lr': [0.005, 0.004, 0.003, 0.002, 0.001],  # 0.003
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [1024],
            'transfer_name': ['e1024'],
            'transfer': [True]
        }

    # 512

    elif config['name'] == 't512fwlin_grid':
        return {
            'lr': [0.03, 0.02, 0.01, 0.009, 0.008],  # 0.02
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True]
        }

    # 300

    elif config['name'] == 't300fwlin_grid':
        return {
            'lr': [0.05, 0.04, 0.03, 0.02, 0.01],  # 0.
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [300],
            'transfer_name': ['e300'],
            'transfer': [True]
        }

    # 100

    elif config['name'] == 't100fwlin_grid':
        return {
            'lr': [0.2, 0.1, 0.09, 0.08, 0.07],  # 0.
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [100],
            'transfer_name': ['e100'],
            'transfer': [True]
        }

    #
    # CompC Model

    # 512

    elif config['name'] == 't512fwcomprw_grid':
        return {
            'lr': [0.2, 0.09, 0.06, 0.03, 0.01, 0.009, 0.006, 0.002, 0.001],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True]
        }

    #
    # CompC Model

    # 512

    elif config['name'] == 't512fwcompc_grid':
        return {
            'lr': [0.005, 0.004, 0.002, 0.001, 0.0009],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'emb_lr_factor': [0.1],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True],
            'hidden_size': [512]
        }

    #
    # CompRW Model

    # 768

    elif config['name'] == 't768fwcomprw_grid':
        return {
            'lr': [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004,  # 0.004
                   0.003, 0.002, 0.001],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [768],
            'transfer_name': ['e768'],
            'transfer': [True]
        }

    # 640

    elif config['name'] == 't640fwcomprw_grid':
        return {
            'lr': [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [640],
            'transfer_name': ['e640'],
            'transfer': [True]
        }

    # 512

    elif config['name'] == 't512fwcomprw_grid':
        return {
            'lr': [0.004, 0.003, 0.002, 0.001, 0.0009],  # 0.01
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True]
        }

    #
    # FROZEN ENCODERS

    elif config['name'] == 't512fecomp_grid':
        return {
            'lr': [0.3, 0.2, 0.1, 0.009, 0.006, 0.003, 0.001],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [False],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True],
            'hidden_size': [512]
        }
    elif config['name'] == 't512felin_grid':
        return {
            'lr': [0.3, 0.2, 0.1, 0.009, 0.006, 0.003, 0.001],
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [False],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True]
        }

    #
    # CompBCE Model

    # 512

    elif config['name'] == 't512fwcompbce_grid':
        return {
            'lr': [0.006, 0.005, 0.004, 0.003, 0.002],  # 0.003
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True],
            'hidden_size': [512]
        }

    #
    # CompMLP Model

    # 512

    elif config['name'] == 't512fwcompmlp_grid':
        return {
            'lr': [0.006, 0.005, 0.004, 0.003, 0.002],  # 0.003
            'l2': [0.],
            'p_drop': [0.1],
            'tune_embeds': [False],
            'tune_encoder': [True],
            'enc_lr_factor': [1.],
            'encoder_size': [512],
            'transfer_name': ['e512'],
            'transfer': [True],
            'hidden_size': [512]
        }

    #
    # Error Checking

    else:
        raise ValueError('Unexpected name %s' % config['name'])
