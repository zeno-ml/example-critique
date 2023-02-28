from zeno import distill, model, metric, ZenoOptions
from inspiredco.critique import Critique
import os
from sentence_transformers import SentenceTransformer

client = Critique(api_key=os.environ["INSPIREDCO_API_KEY"])


@model
def pred_fns(name):
    sentence_embed = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    def pred(df, ops):
        embed = sentence_embed.encode(df[ops.data_column].tolist()).tolist()
        return df["translation"], embed

    return pred


@distill
def bert_score(df, ops):
    eval_dict = df[["source", ops.output_column, "label"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("label")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="bert_score", config={"model": "bert-base-uncased"}, dataset=eval_dict
    )

    return [round(r["value"], 6) for r in result["examples"]]


@metric
def avg_bert_score(df, ops: ZenoOptions):
    return df[ops.distill_columns["bert_score"]].mean()


@distill
def length(df, ops):
    return df[ops.data_column].str.len()
