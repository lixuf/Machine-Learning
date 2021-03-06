class OurLayer(Layer):

    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """

    def reuse(self, layer, *args, **kwargs):

        if not layer.built:

            if len(args) > 0:

                inputs = args[0]

            else:

                inputs = kwargs['inputs']

            if isinstance(inputs, list):

                input_shape = [K.int_shape(x) for x in inputs]

            else:

                input_shape = K.int_shape(inputs)

            layer.build(input_shape)

        outputs = layer.call(*args, **kwargs)

        for w in layer.trainable_weights:

            if w not in self._trainable_weights:

                self._trainable_weights.append(w)

        for w in layer.non_trainable_weights:

            if w not in self._non_trainable_weights:

                self._non_trainable_weights.append(w)

        return outputs
        
        
 class DilatedGatedConv1D(OurLayer):

    """膨胀门卷积（DGCNN）
    """

    def __init__(self,

                 o_dim=None,

                 k_size=3,

                 rate=1,

                 skip_connect=True,

                 drop_gate=None,

                 **kwargs):

        super(DilatedGatedConv1D, self).__init__(**kwargs)

        self.o_dim = o_dim

        self.k_size = k_size

        self.rate = rate

        self.skip_connect = skip_connect

        self.drop_gate = drop_gate

    def build(self, input_shape):

        super(DilatedGatedConv1D, self).build(input_shape)

        if self.o_dim is None:

            self.o_dim = input_shape[0][-1]

        self.conv1d = Conv1D(

            self.o_dim * 2,

            self.k_size,

            dilation_rate=self.rate,

            padding='same'

        )

        if self.skip_connect and self.o_dim != input_shape[0][-1]:

            self.conv1d_1x1 = Conv1D(self.o_dim, 1)

    def call(self, inputs):

        xo, mask = inputs

        x = xo * mask

        x = self.reuse(self.conv1d, x)

        x, g = x[..., :self.o_dim], x[..., self.o_dim:]

        if self.drop_gate is not None:

            g = K.in_train_phase(K.dropout(g, self.drop_gate), g)

        g = K.sigmoid(g)

        if self.skip_connect:

            if self.o_dim != K.int_shape(xo)[-1]:

                xo = self.reuse(self.conv1d_1x1, xo)

            return (xo * (1 - g) + x * g) * mask

        else:

            return x * g * mask

    def compute_output_shape(self, input_shape):

        return input_shape[0][:-1] + (self.o_dim,)
