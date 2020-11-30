from mxnet.gluon.loss import Loss, _reshape_like


class SoftmaxCrossEntropyCustomWeightRFS(Loss):
    """SoftmaxCrossEntropyLoss with ignore labels

    ##### Loss Function is from another project with weighted image object segmentation, adapt it to the
    ##### sound problem to be working

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
        Whether to re-scale loss with regard to ignored labels.
    """

    def __init__(self, batch_axis=0, ignore_label=-1, weight=0.1, **kwargs):
        super(SoftmaxCrossEntropyCustomWeightRFS, self).__init__(None, batch_axis, **kwargs)
        self._ignore_label = ignore_label
        self._weight = weight
        self._sparse_label = False
        self._minDiv = 1E-5
        self._size_average = True

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null')
        if self._sparse_label:
            loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(F.log(softmaxout) * label, axis=-1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        # label = nd.expand_dims(label, axis=1)
        # sum1 = nd.sum(loss*(label+self._weight), axis=(2,3))
        # sum2 = nd.sum((label+self._weight), axis=(2, 3)) + self._minDiv
        # weight_loss = sum1/sum2
        return loss
