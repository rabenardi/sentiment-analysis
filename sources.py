from util.sanitizer import Sanitizer
from util.source_reader import Source
from os import path
import constant

# Menyiapkan metadata
samples = {
    "source": Source(
        name="Samples Main",
        the_path=path.join(path.dirname(__file__), "samples"),
        log=True # Berisik cak
    ),
    "sentiment_col": "Sentiment",
    "text_col": "Text",
}

# menyiapkan sampel pelatihan
samples["source"]\
.prepare_for_sentiment_analysis(
    sentiment_col=samples["sentiment_col"], 
    text_col=samples["text_col"],
    f_sanitize=Sanitizer.sanitize
)

source_neutral = Source(
    the_path=path.join(path.dirname(__file__), "samples", "neutral-sample"),
    name="samples neutral",
    log=False
)

# import numpy as np
# import collections
# dataframe = samples["source"].dataframes
# print(samples["source"].sentiment_values, collections.Counter(dataframe[0]))

# Memasukkan sampel pelatihan lain jika ada
other_samples = [
    {
        "path": path.join(path.dirname(__file__), "samples", "neutral-sample"),
        "filter_equal": [samples["sentiment_col"], "neutral"]
    }
]

if len(other_samples):
    for sample in other_samples:
        tmp = Source(name="Dummy", the_path=sample["path"], log=False)
        tmp.fetch().flatten()
        
        if "filter_equal" in sample:
            l_filter = sample["filter_equal"]
            tmp.dataframes = tmp.dataframes[tmp.dataframes[l_filter[0]] == l_filter[1]]

        tmp.normalize_for_sentiment_analysis(
            sentiment_col=samples["sentiment_col"], 
            text_col=samples["text_col"]
        )
        
        samples["source"].join_for_sentimental_analysis(tmp)
        del tmp

# print(samples["source"].sentiment_values, collections.Counter(dataframe[0]))