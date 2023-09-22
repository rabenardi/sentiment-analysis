from util.sanitizer import Sanitizer
from util.source_reader import Source
from os import path

samples = {
    "source": Source(
        name="Samples Main",
        the_path=path.join(path.dirname(__file__), "sample"),
        log=False #Berisik cak
    ),
    "sentiment_col": "Sentiment",
    "text_col": "Text",
}

samples["source"].fetch(depth=float("inf"))
samples["source"].normalize_for_sentiment_analysis(sentiment_col=samples["sentiment_col"], text_col=samples["text_col"])
samples["source"].map(1, Sanitizer.sanitize)

other_samples = []

if len(other_samples) > 1:
    for sample in other_samples:
        tmp = Source(name="Dummy", the_path=sample.path)
        tmp.fetch()
        tmp.normalize(
            sentiment_col=sample["sentiment_col"], 
            text_col=sample["text_col"]
        )

        samples["source"].join(tmp)
        del tmp
