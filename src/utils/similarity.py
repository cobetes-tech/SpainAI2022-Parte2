import numpy as np
import sklearn
import pandas as pd
import bokeh
import bokeh.models
import bokeh.plotting

def _filter_threshold(x, threshold=0):
    return ((x > threshold) & (x < 1)).any()


def similarity_filter(embeddings_1, embeddings_2, labels_1, labels_2, threshold=0):
    sim = sklearn.metrics.pairwise.cosine_similarity(embeddings_1, embeddings_2)
    np.fill_diagonal(sim, 1)

    tmp = np.apply_along_axis(_filter_threshold, axis=1, arr=sim, threshold=threshold)
    sim = np.delete(sim, ~tmp, axis=0)
    sim = np.delete(sim, ~tmp, axis=1)
    labels_1 = list(map(labels_1.__getitem__, np.where(tmp)[0]))
    labels_2 = list(map(labels_2.__getitem__, np.where(tmp)[0]))
    embeddings_1 = list(map(embeddings_1.__getitem__, np.where(tmp)[0]))
    embeddings_2 = list(map(embeddings_2.__getitem__, np.where(tmp)[0]))

    embeddings_1_col = list()
    embeddings_2_col = list()
    sim_col = list()
    for i in range(len(embeddings_1)):
        for j in range(len(embeddings_2)):
            embeddings_1_col.append(labels_1[i])
            embeddings_2_col.append(labels_2[j])
            sim_col.append(sim[i][j])
    df = pd.DataFrame(
        zip(embeddings_1_col, embeddings_2_col, sim_col),
        columns=["embeddings_1", "embeddings_2", "sim"],
    )

    return df, labels_1, labels_2

def visualize_similarity(
    embeddings_1,
    embeddings_2,
    labels_1,
    labels_2,
    plot_title,
    threshold=0,
    plot_width=1200,
    plot_height=600,
    xaxis_font_size="8pt",
    yaxis_font_size="8pt",
):

    assert len(embeddings_1) == len(labels_1)
    assert len(embeddings_2) == len(labels_2)

    df, labels_1, labels_2 = similarity_filter(
        embeddings_1, embeddings_2, labels_1, labels_2, threshold
    )

    mapper = bokeh.models.LinearColorMapper(
        palette=[*reversed(bokeh.palettes.YlOrRd[9])],
        low=df.sim.min(),
        high=df.sim.max(),
    )

    p = bokeh.plotting.figure(
        title=plot_title,
        x_range=labels_1,
        x_axis_location="above",
        y_range=[*reversed(labels_2)],
        plot_width=plot_width,
        plot_height=plot_height,
        tools="save",
        toolbar_location="below",
        tooltips=[("pair", "@embeddings_1 ||| @embeddings_2"), ("sim", "@sim")],
    )
    p.rect(
        x="embeddings_1",
        y="embeddings_2",
        width=1,
        height=1,
        source=df,
        fill_color={"field": "sim", "transform": mapper},
        line_color=None,
    )

    p.title.text_font_size = "12pt"
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 16
    p.xaxis.major_label_text_font_size = xaxis_font_size
    p.xaxis.major_label_orientation = 0.25 * np.pi
    p.yaxis.major_label_text_font_size = yaxis_font_size
    p.min_border_right = 300

    return p
