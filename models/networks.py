import models.modules.CDic_Align_s as CDic_Align_s
import models.modules.CDic_Align_l as CDic_Align_l


####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'A2-CDic-s':
        netG = CDic_Align_s.CDic_Align()
    elif which_model == 'A2-CDic-l':
        netG = CDic_Align_l.CDic_Align()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
