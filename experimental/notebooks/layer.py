""" Layer definition """

STD = "standard"
DPT = "depthwise"
PNT = "pointwise"


class BaseLayer:
  def __init__(self, name):
    self.name = name

  @property
  def num_params(self):
    pass

  @property
  def num_ops(self):
    pass


class ConvLayer(BaseLayer):
  def __init__(self, name, H, W, C, F, K, T):
    super().__init__(name)
    self.H = H
    self.W = W
    self.C = C
    self.F = F
    self.K = K
    self.T = T  # type

  def __str__(self):
    return "%10s: <%3d, %3d, %4d, %4d, %d, %10s>" \
        % (self.name, self.H, self.W, self.C, self.F, self.K, self.T)

  def __repr__(self):
    return self.__str__()

  @property
  def num_params(self):
    return self.C * self.F * self.K * self.K

  @property
  def num_ops(self):
    return 2 * self.C * self.F * self.H * self.W * self.K * self.K


class MatmulLayer(BaseLayer):
  def __init__(self, name, H, W):
    super().__init__(name)
    self.name = name
    self.num_rows = H
    self.num_cols = W

  def __str__(self):
    return "%10s: <%4d, %4d>" % (self.name, self.num_rows, self.num_cols)

  def __repr__(self):
    return self.__str__()

  @property
  def num_params(self):
    return self.num_cols * self.num_rows

  @property
  def num_ops(self):
    return 2 * self.num_cols * self.num_rows
