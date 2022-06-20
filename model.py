import torch
from torch import nn

from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet
from models.pytorch_i3d import InceptionI3d
from models.tea_model import tea50_16f
from models.YMAttentionNetwork import SwinTransformer_AT_1xbi_lstm, SwinTransformer_AT
from models.Swim_R2p1d import SwinTransformer_R2plus1d, SwinTransformer_TwoStream, SwinTransformer_OneStream, vitbase
from models.STR_transformer import STR_Transformer


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'i3d', "tea",
        "SwinTransformer_AT_1xbi_lstm", "SwinTransformer_OneStream", "SwinTransformer_R2plus1d",
        "SwinTransformer_TwoStream", "SwinTransformer_AT", "STR_Transformer", "vitbase"
    ]

    if opt.model == 'i3d':
        model = InceptionI3d(opt.n_classes, in_channels=3)

    elif opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt.model_depth,
                                          n_classes=opt.n_classes,
                                          n_input_channels=opt.n_input_channels,
                                          shortcut_type=opt.resnet_shortcut,
                                          conv1_t_size=opt.conv1_t_size,
                                          conv1_t_stride=opt.conv1_t_stride,
                                          no_max_pool=opt.no_max_pool,
                                          widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'wideresnet':
        model = wide_resnet.resnet50(
            k=opt.wide_resnet_k,
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'resnext':
        model = resnext.generate_model(model_depth=opt.model_depth,
                                       cardinality=opt.resnext_cardinality,
                                       n_classes=opt.n_classes,
                                       n_input_channels=opt.n_input_channels,
                                       shortcut_type=opt.resnet_shortcut,
                                       conv1_t_size=opt.conv1_t_size,
                                       conv1_t_stride=opt.conv1_t_stride,
                                       no_max_pool=opt.no_max_pool)
    elif opt.model == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'densenet':
        model = densenet.generate_model(model_depth=opt.model_depth,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)
    elif opt.model == "tea":
        model = tea50_16f(pretrained=False, num_classes=opt.n_classes)
    elif opt.model == "SwinTransformer_OneStream":
        model = SwinTransformer_OneStream(at_type=opt.at_type, num_classes=opt.n_classes, lstm_channel=opt.num_frames)
    elif opt.model == "SwinTransformer_AT_1xbi_lstm":
        model = SwinTransformer_AT_1xbi_lstm(at_type=opt.at_type, num_classes=opt.n_classes, lstm_channel=opt.num_frames)
    elif opt.model == "SwinTransformer_R2plus1d":
        model = SwinTransformer_R2plus1d(at_type=opt.at_type, num_classes=opt.n_classes, lstm_channel=opt.num_frames)
    elif opt.model == "SwinTransformer_TwoStream":
        model = SwinTransformer_TwoStream(at_type=opt.at_type, num_classes=opt.n_classes)
    elif opt.model == "SwinTransformer_AT":
        model = SwinTransformer_AT(num_classes=opt.n_classes)
    elif opt.model == "STR_Transformer":
        model = STR_Transformer(at_type=opt.at_type, num_classes=opt.n_classes, lstm_channel=opt.num_frames)
    elif opt.model == "vitbase":
        model = vitbase(num_classes=opt.n_classes)

    return model


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        if model_name == "SwinTransformer_TwoStream":
            print('loading checkpoint {} model'.format(pretrain_path))
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])

            return model

        elif model_name == 'i3d':
            model.replace_logits(157)
            model.load_state_dict(torch.load(pretrain_path))
            model.replace_logits(n_finetune_classes)
        else:
            print('loading pretrained model {}'.format(pretrain_path))
            pretrain = torch.load(pretrain_path, map_location='cpu')

            if model_name == 'wideresnet':
                pretrain2 = {'state_dict': {}}
                for k, v in pretrain['state_dict'].items():
                    new_name = k[7:]
                    pretrain2['state_dict'][new_name] = v
                pretrain = pretrain2
            model.load_state_dict(pretrain['state_dict'])
            tmp_model = model

            if model_name == 'densenet':
                tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                                 n_finetune_classes)
            else:
                tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                         n_finetune_classes)

    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
