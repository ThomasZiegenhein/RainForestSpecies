from mxnet import initializer
from mxnet.gluon import nn

from src.helpfun.constants import NUM_SPECIES


class ResNet181DLib:
    """"
    Contains the standard bricks for the ResNet18 model in 1D. Naming convention see:
    https://arxiv.org/pdf/1512.03385.pdf
    """

    @staticmethod
    def get_conv2():
        body = nn.HybridSequential(prefix='conv2')
        for i in range(2):
            body.add(ResNetBlock1D(chn=64))
        return body

    @staticmethod
    def get_conv3():
        body = nn.HybridSequential(prefix='conv3')
        body.add(FirstResNetBlock1D(chn=128))
        for i in range(1):
            body.add(ResNetBlock1D(chn=128))
        return body

    @staticmethod
    def get_conv4():
        body = nn.HybridSequential(prefix='conv4')
        body.add(FirstResNetBlock1D(chn=256))
        for i in range(1):
            body.add(ResNetBlock1D(chn=256))
        return body

    @staticmethod
    def get_conv5():
        body = nn.HybridSequential(prefix='conv5')
        body.add(FirstResNetBlock1D(chn=512))
        for i in range(1):
            body.add(ResNetBlock1D(chn=512))
        return body


class FirstResNetBlock1D(nn.HybridBlock):
    """ResNet block with down sampling"""
    def __init__(self, chn, **kwargs):
        super(FirstResNetBlock1D, self).__init__(**kwargs)
        self.chn = chn
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv1D(channels=self.chn, kernel_size=3, strides=2, padding=1, dilation=1)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv1D(channels=self.chn, kernel_size=3, strides=1, padding=1, dilation=1)
            self.conv_down_sample = nn.Conv1D(channels=self.chn, kernel_size=1, strides=2, padding=0, dilation=1)
            self.max_pool = nn.MaxPool1D(pool_size=5, strides=5, padding=2)

    def hybrid_forward(self, F, x):
        res = self.conv_down_sample(x)
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)
        x = x + res
        x = self.max_pool(x)
        return x


class ResNetBlock1D(nn.HybridBlock):
    """Standard ResNet block"""
    def __init__(self, chn, **kwargs):
        super(ResNetBlock1D, self).__init__(**kwargs)
        self.chn = chn
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv1D(channels=self.chn, kernel_size=3, strides=1, padding=1, dilation=1)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv1D(channels=self.chn, kernel_size=3, strides=1, padding=1, dilation=1)

    def hybrid_forward(self, F, x):
        res = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)
        return x + res

class ResNet18RFSV0(nn.HybridBlock):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition"
    ` https://arxiv.org/pdf/1512.03385.pdf adopted with ideas from
    ` https://arxiv.org/pdf/1705.09759.pdf
    """

    def __init__(self, **kwargs):
        super(ResNet18RFSV0, self).__init__(**kwargs)
        with self.name_scope():
            self.base = nn.HybridSequential(prefix='')
            self.conv2 = nn.HybridSequential(prefix='conv2')
            self.conv3 = nn.HybridSequential(prefix='conv3')
            self.conv4 = nn.HybridSequential(prefix='conv4')
            self.conv5 = nn.HybridSequential(prefix='conv5')
            self.avg_pool = nn.AvgPool1D(pool_size=9)

            self.base.add(nn.Conv1D(channels=64, kernel_size=7, strides=1, padding=3, dilation=1, use_bias=False,
                                    in_channels=1))
            self.base.add(nn.BatchNorm())
            self.base.add(nn.Activation('relu'))
            self.conv2.add(ResNet181DLib.get_conv2())
            self.conv3.add(ResNet181DLib.get_conv3())
            self.conv4.add(ResNet181DLib.get_conv4())
            self.conv5.add(ResNet181DLib.get_conv5())

    def hybrid_forward(self, F, x):
        xbase = self.base(x)
        xconv2 = self.conv2(xbase)
        xconv3 = self.conv3(xconv2)
        xconv4 = self.conv4(xconv3)
        xconv5 = self.conv5(xconv4)
        x = self.avg_pool(xconv5)
        return x


class ResNetHeadV0(nn.HybridBlock):
    """
    Head to classify the parameters extracted from the ResNet model. V0: standard MLP

    Parameters
    ----------
    :param num_input : int, default None (defined at runtime)
    :param num_classes : int, default 26 (NUM_SPECIES+1)
    :param Number of classification classes.
    """
    def __init__(self, num_input=None, num_classes=NUM_SPECIES+1, **kwargs):
        super(ResNetHeadV0, self).__init__(**kwargs)
        self.num_input = num_input
        self.num_classes = num_classes
        with self.name_scope():
            if num_input is None:
                self.first_layer = nn.Dense(units=num_classes)
            else:
                self.first_layer = nn.Dense(units=num_classes, in_units=num_input)

    def hybrid_forward(self, F, x):
        x = self.first_layer(x)
        return x


class StandardBodyHead(nn.HybridBlock):
    """Combines the feature extractor (body) and the feature classifier (head)"""

    def __init__(self, body, head, **kwargs):
        super(StandardBodyHead, self).__init__(**kwargs)
        self.body = body
        self.head = head

    def hybrid_forward(self, F, x):
        x = self.body(x)
        x = self.head(x)
        return x


class FactoryModels:
    """Factory to create neural networks for the problem"""
    @staticmethod
    def get_standard_model():
        body = ResNet18RFSV0()
        head = ResNetHeadV0()
        model = StandardBodyHead(body=body, head=head)
        model.initialize(init=initializer.Xavier())
        return model

