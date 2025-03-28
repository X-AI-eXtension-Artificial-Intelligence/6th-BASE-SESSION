import torch.nn as nn
import torch.utils.model_zoo as model_zoo  # 미리 학습된 가중치값을 가지는 모델을 불러오는 모듈 


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):  # conv layer, 분류할 클래스 1000개, 가중치 초기화 여부 
        super(VGG, self).__init__()

        self.features = features  # 우리가 쌓아야하는 convoluion layer들 

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 입력을 7x7로 고정하는 pooling layer 

        self.classifier = nn.Sequential(  # FC layer. 최종 분류기 
            # nn.Linear(512 * 7 * 7, 4096),  # oroginal 
            nn.Linear(1024 * 7 * 7, 4096),  # transformed_model 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):  # 순전파 
        x = self.features(x) 
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)  # flattening
        x = self.classifier(x)  # 분류기 
        return x

    def _initialize_weights(self):
        for m in self.modules():  # nn.Module 클래스의 내장 함수. 
                                  # 모델 내의 모든 서브모듈(submodules)을 반환하는 반복 가능(iterable) 객체
                                  # 모델을 구성하는 모든 nn.Module 레이어를 하나씩 가져옴.
            if isinstance(m, nn.Conv2d):   # convolution layer이면 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He 초기화 적용 
                if m.bias is not None:  
                    nn.init.constant_(m.bias, 0)  # 편향이 존재하면 0으로 초기화 

            elif isinstance(m, nn.BatchNorm2d):  # BN layer이면
                nn.init.constant_(m.weight, 1)  # 가중치 1로 초기화
                nn.init.constant_(m.bias, 0)  # 편향 0으로 초기화 

            elif isinstance(m, nn.Linear):  
                nn.init.normal_(m.weight, 0, 0.01)  # 평균=0, std=0.01 정규분포를 이용해 초기화
                nn.init.constant_(m.bias, 0)  # 편향 0으로 초기화 


def make_layers(cfg, batch_norm=False):  # vgg class의 features에 들어갈 layer 만드는 층 
    layers = []
    in_channels = 3
    for v in cfg:  # 리스트 요소들을 불러옴. 요소가 숫자/'M' 으로 구성된 리스트
        if v == 'M': 
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 2x2크기 필터 추가. 
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # convolution layer 정의. 입력채널수, 출력채널수, 필터크기, 패딩 
            if batch_norm:  # BN층 쓸것인가? 하이퍼파라미터 따라 결정 
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]  # conv, BN, RelU layer 추가 
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]  # conv, ReLU만 추가 
            in_channels = v  # 출력채널수가 다음번 입력채널 수 
    return nn.Sequential(*layers)  # 여러 nn.Module을 순서대로 실행하는 클래스 


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model



def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model



def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model



def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model



def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model



def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model



def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model



def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model