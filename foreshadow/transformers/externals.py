"""External Transformers

All sklearn transformers imported here will be wrapped and made available in the
module foreshadow.transformers

"""

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Imputer
from sklearn.decomposition import PCA
from category_encoders import OneHotEncoder, HashingEncoder
