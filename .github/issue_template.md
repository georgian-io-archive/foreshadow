#### Description
<!-- 
Example: DropFeatures fails on categorical features when assessing a string column
-->

#### Steps/Code to Reproduce
<!--
Please add the minimum code required to reproduce the issue if possible.
Example:
```python
import uuid
import numpy as np
import pandas as pd
from foreshadow.preprocessor import Preprocessor

cat1 = [str(uuid.uuid4()) for _ in range(40)]
cat2 = [str(uuid.uuid4()) for _ in range(40)]

input = pd.DataFrame({
    'col1': np.random.choice(cat1, 1000),
    'col2': np.random.choice(cat2, 1000)
})

processor = Preprocessor()
output = processor.fit_transform(input)
```
If the code is too long, feel free to put it in a public gist and link it in the issue: https://gist.github.com
-->

#### Expected Results
<!--
Please add the results that you would expect here.
Example: Error should not be thrown
-->

#### Actual Results
<!--
Please place the full traceback here, again use a gist if you feel that it is too long.
-->

#### Versions
<!--
Please run the following snippet in your environment and paste the results here.

```python
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import pandas; print("Pandas", pandas.__version__)
import foreshadow; print("Foreshadow", foreshadow.__version__)
from foreshadow.utils import check_transformer_imports; check_transformer_imports()
```
-->


<!--Thank you for contributing!-->
