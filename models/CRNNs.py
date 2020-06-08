import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utilities import ConvBlock, init_gru, init_layer, interpolate


class CRNN9(nn.Module):
    #def __init__(self, class_num, pool_type='avg', pool_size=(2,2), pretrained_path=None):
    def __init__(self, class_num, pretrained_path=None, args=None):

        super().__init__()

        self.class_num = class_num
        self.pool_type = args.model_pool_type
        self.pool_size = args.model_pool_size
        self.interp_ratio = 8
        inp_dict = {'logmel':7, 'logmelgcc':7, 'logmelintensity':7, 'logmelgccintensity':17}
        self.inp_chs = inp_dict[args.feature_type]
        
        self.conv_block1 = ConvBlock(in_channels=inp_dict, out_channels=128)    # 1: 7, 128     2: 7, 64
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=256)  # 1: 128, 256   2: 64, 256
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(args.model_gru_size*2 , class_num, bias=True) # *2 : because GRU is bidirectional
        self.azimuth_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)
        self.elevation_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        print("tmp =========> ")

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        # self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)     
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        elevation_output = interpolate(elevation_output, self.interp_ratio)
        
        output = {
            'events': event_output,
            'doas': torch.cat((azimuth_output, elevation_output), dim=-1)
        }

        return output


class pretrained_CRNN8(CRNN9):

    def __init__(self, class_num, pretrained_path=None, args=None):
        self.args = args
        super().__init__(class_num, pretrained_path=pretrained_path, args=args)
        
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)   

    def load_weights(self, pretrained_path):

        model = CRNN9(self.class_num, args=self.args)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3


class CRNN9_logmelgccintensity(nn.Module):
    def __init__(self, class_num, pretrained_path=None, args=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = args.model_pool_type
        self.pool_size = args.model_pool_size
        self.interp_ratio = 8
        
        self.conv_block1 = ConvBlock(in_channels=17, out_channels=128)    # 1: 7, 128     2: 7, 64
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=256)  # 1: 128, 256   2: 64, 256
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(args.model_gru_size*2 , class_num, bias=True) # *2 : because GRU is bidirectional
        self.azimuth_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)
        self.elevation_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        # self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)     
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        elevation_output = interpolate(elevation_output, self.interp_ratio)
        
        output = {
            'events': event_output,
            'doas': torch.cat((azimuth_output, elevation_output), dim=-1)
        }

        return output


class pretrained_CRNN8_logmelgccintensity(CRNN9_logmelgccintensity):

    def __init__(self, class_num, pretrained_path=None, args=None):
        self.args = args
        super().__init__(class_num, pretrained_path=pretrained_path, args=args)
        
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)   

    def load_weights(self, pretrained_path):

        # 1. orgin #
        # model = CRNN9_logmelgccintensity(self.class_num, args=self.args)
        # checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        # model.load_state_dict(checkpoint['model_state_dict'])
        #
        # self.conv_block1 = model.conv_block1
        # self.conv_block2 = model.conv_block2
        # self.conv_block3 = model.conv_block3

        # 2. second trial #
        # model = CRNN9_logmelgccintensity(self.class_num, args=self.args)
        # checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        # model.conv_block1.load_state_dict(checkpoint['model_state_dict'].conv_block1)
        # model.conv_block2.load_state_dict(checkpoint['model_state_dict'].conv_block2)
        # model.conv_block3.load_state_dict(checkpoint['model_state_dict'].conv_block3)
        #
        # self.conv_block1 = model.conv_block1
        # self.conv_block2 = model.conv_block2
        # self.conv_block3 = model.conv_block3

        # 3. third trial #
        model = CRNN9_logmelgccintensity(self.class_num, args=self.args)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)

        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        # filter out unnecessary keys #
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if (k in model_dict) and
                           (v.size()==model_dict[k].size()) and
                           (("gru" not in k) or ("_fc" not in k))
                           }
        # overwrite entries in the existing state dict #
        model_dict.update(pretrained_dict)
        # load the new state dict #
        self.load_state_dict(model_dict)

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        

class CRNN11(nn.Module):
    def __init__(self, class_num, pretrained_path=None, args=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = args.model_pool_type
        self.pool_size = args.model_pool_size
        self.interp_ratio = 16
        
        self.conv_block1 = ConvBlock(in_channels=7, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(args.model_gru_size*2 , class_num, bias=True) # *2 : because GRU is bidirectional
        self.azimuth_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)
        self.elevation_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        # self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)     
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        elevation_output = interpolate(elevation_output, self.interp_ratio)
        
        output = {
            'events': event_output,
            'doas': torch.cat((azimuth_output, elevation_output), dim=-1)
        }

        return output


class pretrained_CRNN10(CRNN11):

    def __init__(self, class_num, pretrained_path=None, args=None):
        self.args = args
        super().__init__(class_num, pretrained_path=pretrained_path, args=args)
        
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)   

    def load_weights(self, pretrained_path):

        model = CRNN11(self.class_num, args=self.args)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4


class CRNN11_logmelgccintensity(nn.Module):
    def __init__(self, class_num, pretrained_path=None, args=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = args.model_pool_type
        self.pool_size = args.model_pool_size
        self.interp_ratio = 16
        
        self.conv_block1 = ConvBlock(in_channels=17, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(args.model_gru_size*2 , class_num, bias=True) # *2 : because GRU is bidirectional
        self.azimuth_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)
        self.elevation_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        # self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)     
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        elevation_output = interpolate(elevation_output, self.interp_ratio)
        
        output = {
            'events': event_output,
            'doas': torch.cat((azimuth_output, elevation_output), dim=-1)
        }

        return output


class pretrained_CRNN10_logmelgccintensity(CRNN11_logmelgccintensity):

    def __init__(self, class_num, pretrained_path=None, args=None):
        self.args = args
        super().__init__(class_num, pretrained_path=pretrained_path, args=args)
        
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)   

    def load_weights(self, pretrained_path):

        model = CRNN11_logmelgccintensity(self.class_num, args=self.args)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4


class Gated_CRNN9(nn.Module):
    def __init__(self, class_num, pretrained_path=None, args=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = args.model_pool_type
        self.pool_size = args.model_pool_size
        self.interp_ratio = 8
        
        self.conv_block1 = ConvBlock(in_channels=7, out_channels=64)    # 1: 7, 128     2: 7, 64
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=256)  # 1: 128, 256   2: 64, 256
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=512)

        self.gate_block1 = ConvBlock(in_channels=7, out_channels=64)
        self.gate_block2 = ConvBlock(in_channels=64, out_channels=256)
        self.gate_block3 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)

        self.event_fc = nn.Linear(args.model_gru_size*2 , class_num, bias=True) # *2 : because GRU is bidirectional
        self.azimuth_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)
        self.elevation_fc = nn.Linear(args.model_gru_size*2, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        gate = self.gate_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)

        gate = self.gate_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)

        gate = self.gate_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = x * torch.sigmoid(gate)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1,2)
        ''' (batch_size, time_steps, feature_maps):'''

        # self.gru.flatten_parameters()
        (x, _) = self.gru(x)
        
        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)     
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        elevation_output = interpolate(elevation_output, self.interp_ratio)
        
        output = {
            'events': event_output,
            'doas': torch.cat((azimuth_output, elevation_output), dim=-1)
        }

        return output


class pretrained_Gated_CRNN8(Gated_CRNN9):

    def __init__(self, class_num, pretrained_path=None, args=None):
        self.args= args
        super().__init__(class_num, pretrained_path=pretrained_path, args=args)
        
        if pretrained_path:
            self.load_weights(pretrained_path)

        self.gru = nn.GRU(input_size=512, hidden_size=args.model_gru_size,
            num_layers=1, batch_first=True, bidirectional=True)

        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)   

    def load_weights(self, pretrained_path):

        model = Gated_CRNN9(self.class_num, args=self.args)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3