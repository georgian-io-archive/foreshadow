"""Classes for optimizing Foreshadow given a param_distribution."""
import foreshadow as fs
import foreshadow.serializers as ser



"""
combinations:
    X_preparer.cleaner.CHAS:
        Cleaner:
            - date:
                - p1
                - p2
            - financial
        IntentMapper:
            - Something

    X_preparer.cleaner.CHAS.CleanerMapper:
        -Something

    X_preparer.cleaner.CHAS.IntentMapper:
        -Something


    X_preparer:
        cleaner:
            CHAS:
                Cleaner:
                    date:
                        -p1
                        -p2
                        
                        
Convention:
    Column name is last. If a .<blank> is present, then applied across all 
    columns.

Things that may be swapped:
    PreparerSteps,
    StepSmartTransformers/ConcreteTransformers.

"""


