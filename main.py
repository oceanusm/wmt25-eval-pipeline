"""
    Adaptation of translation collection repository.
    Adapter: Liu Martin

    Adapted from https://github.com/wmt-conference/wmt-collect-translations
    Original Title: Machine Translation API command line tool for translating WMT testsets
    Original Author: Tom Kocmi
"""

import os
import pandas as pd
from absl import flags, app
from tools.utils import collect_answers, SYSTEMS
 

flags.DEFINE_enum('system', 'GPT-OSS-20B', SYSTEMS, 'Define the system to use for translation')

FLAGS = flags.FLAGS

def main(args):
    assert os.path.exists("wmt25-genmt.jsonl"), "File downloaded from github. Kept with the evaluation code."
    blindset = pd.read_json("wmt25-genmt.jsonl", lines=True)

    answers = collect_answers(blindset, FLAGS.system)
    df = pd.DataFrame(answers)

    # for each tgt_lang, count how many "FAILED" there are and if more than 25% are FAILED, remove that tgt_lang
    for tgt_lang in df['tgt_lang'].unique():
        num_none = df[df['tgt_lang'] == tgt_lang]['hypothesis'].str.contains("FAILED", na=False).sum()
        if num_none > 0.25 * len(df[df['tgt_lang'] == tgt_lang]):
            df = df[df['tgt_lang'] != tgt_lang]

    os.makedirs("wmt_translations", exist_ok=True)
    df.to_json(f"wmt_translations/{FLAGS.system}.jsonl", orient='records', lines=True, force_ascii=False)

    mt_num_none = df[df['hypothesis'].str.contains("FAILED", na=False)]['hypothesis'].count()
    print(f"Number of untranslated answers in MT: {mt_num_none}")


if __name__ == '__main__':
    app.run(main)