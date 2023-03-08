import os

import numpy as np
import pandas as pd
from inspiredco.critique import Critique

from zeno import (
    DistillReturn,
    MetricReturn,
    ModelReturn,
    ZenoOptions,
    distill,
    metric,
    model,
)

client = Critique(api_key=os.environ["INSPIREDCO_API_KEY"])


@model
def pred_fns(name):
    def pred(df, ops):
        model_df = pd.read_csv(
            ops.label_path + "/{}.tsv".format(name),
            sep="\t",
            quoting=3,
            keep_default_na=False,
        )
        embed_df = pd.read_csv(
            ops.label_path + "/ref_embed.tsv",
            sep="\t",
            keep_default_na=False,
            quoting=3,
        )
        df_join = df[["text"]].merge(
            model_df[["text", "translation"]], on="text", how="left"
        )
        df_join = df_join.merge(embed_df, on="text", how="left")
        return ModelReturn(
            model_output=df_join["translation"].fillna(""),
            embedding=[np.fromstring(d[1:-1], sep=",") for d in df_join["embed"]],
        )

    return pred


@distill
def bert_score(df, ops):
    eval_dict = df[["source", ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="bert_score", config={"model": "bert-base-uncased"}, dataset=eval_dict
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def bleu(df, ops):
    eval_dict = df[[ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="bleu",
        config={"smooth_method": "add_k", "smooth-value": 1.0},
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def chrf(df, ops):
    eval_dict = df[[ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="chrf",
        config={},
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def length_ratio(df, ops):
    eval_dict = df[[ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="length_ratio",
        config={},
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@metric
def avg_bert_score(df, ops: ZenoOptions):
    mean = df[ops.distill_columns["bert_score"]].mean()
    if pd.notna(mean):
        return MetricReturn(metric=mean)
    else:
        return MetricReturn(metric=0)


@metric
def avg_bleu(df, ops: ZenoOptions):
    mean = df[ops.distill_columns["bleu"]].mean()
    if pd.notna(mean):
        return MetricReturn(metric=mean)
    else:
        return MetricReturn(metric=0)


@metric
def avg_chrf(df, ops: ZenoOptions):
    mean = df[ops.distill_columns["chrf"]].mean()
    if pd.notna(mean):
        return MetricReturn(metric=mean)
    else:
        return MetricReturn(metric=0)


@metric
def avg_length_ratio(df, ops: ZenoOptions):
    mean = df[ops.distill_columns["length_ratio"]].mean()
    if pd.notna(mean):
        return MetricReturn(metric=mean)
    else:
        return MetricReturn(metric=0)


@distill
def length(df, ops):
    return DistillReturn(distill_output=df[ops.data_column].str.len())
