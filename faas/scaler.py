from typing import Dict, Optional, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def get_mean_std(
    df: DataFrame, column: str, group_column: Optional[str] = None
) -> Dict[str: Tuple[float, float]]:
    """Return {group_value: (mean, std)} of df[column] grouped by group_column (otherwise 'all').
    """
    if group_column is not None:
        df = df.groupBy(group_column)
    mean_stddevs = df.agg(
        F.mean(F.col(column)).alias('mean'),
        F.stddev(F.col(column)).alias('stddev'),
    ).withColumn(
        'stddev',
        F.when(F.col('stddev').isNull(), F.lit(1.)).otherwise(F.col('stddev'))
    ).collect()
    if group_column is not None:
        return {
            getattr(row, group_column): (getattr(row, 'mean'), getattr(row, 'stddev'))
            for row in mean_stddevs
        }
    else:
        assert len(mean_stddevs) == 1, str(mean_stddevs)
        row = mean_stddevs[0]
        return {'all': (getattr(row, 'mean'), getattr(row, 'stddev'))}
