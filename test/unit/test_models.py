from src.helpfun.constants import NUM_SPECIES


def test_resnet18():
    from src.models.resnet import ResNet18RFSV0
    from mxnet import nd
    from mxnet.gluon import nn
    data = nd.ones((1, 1, 1024 * 8 + 1))
    net = ResNet18RFSV0()
    net.initialize()
    print(net)
    assert nd.flatten(net(data)).shape == nd.ones((1, 512)).shape


def test_head():
    from src.models.resnet import ResNetHeadV0
    from mxnet import nd
    import mxnet.gluon.loss as gloss
    from src.training.loss_fun import SoftmaxCrossEntropyCustomWeightRFS
    data = nd.ones((1, 512))
    net = ResNetHeadV0(num_input=512)
    net.initialize()
    print(net)
    pred = net(data)
    # pred = nd.zeros((1,26))
    # pred[0,25] = 1
    # label = nd.zeros(26)
    # label[25] = 1
    # loss_fun = SoftmaxCrossEntropyCustomWeightRFS()
    # loss_fun = gloss.SoftmaxCrossEntropyLoss(sparse_label= False)
    # print(pred)
    # print(label)
    # print(nd.softmax(pred, axis=1)-label)
    # print(loss_fun(pred,label))
    # print(nd.log(label))
    # softmaxout = F.SoftmaxOutput(
    #    pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
    #    multi_output=self._sparse_label,
    #    use_ignore=True, normalization='valid' if self._size_average else 'null')
    assert (net(data)).shape == nd.ones((1, NUM_SPECIES + 1)).shape


def test_model():
    from src.models.resnet import FactoryModels
    from mxnet import nd
    net = FactoryModels.get_standard_model()
    data = nd.ones((1, 1, 1024 * 8 + 1))
    print(net)
    pred = net(data)
    print(pred)
    assert (net(data)).shape == nd.ones((1, NUM_SPECIES + 1)).shape


test_resnet18()
test_head()
test_model()
