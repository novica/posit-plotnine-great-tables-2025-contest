# Copyright 2024 Marimo. All rights reserved.

import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # What is this notebook about

    I decided to give a shot at [Posit's 2025 Table and Plotnine Contests](https://posit.co/blog/announcing-the-2025-table-and-plotnine-contests/). I also wanted to try `marimo` and `polars` in the process. Initially I wanted this to be a `html-wasm` notebook, but it turned out there are some complications with using `pyarro` in `wasm` that I could not figure out how to resolve, so instead this is a static notebook.

    I was thinking about what data to visualize, and from partly participating in [C Study Group for R Contributors 2025](the https://contributor.r-project.org/events/c-study-group-2025/) I had the R source code on my computer, so it seemed like a good idea to play with the `commits` data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Getting the data

    I have a clone of the repository at [GitHub](https://github.com/wch/r-source). I created a `csv` file with:

    ```
    git log --pretty=format:"%an | %ad | %s" --date=short > ~/r-commits.txt
    ```

    I was trying to figure out what is the best separator, because I thought there will be a lot of comas or semicolons in the commit messages. And it turns out there are a lot of pipes as well. 

    I could not figure out how to import this `csv` with `polars`. But it seems `readr` is much more flexible when importing weird data, and this (reverting to R 🫣): 

    ```
    r_commits <- readr::read_delim("r-commits.txt", 
                            delim = "|", 
                            escape_double = FALSE,
                            col_names = FALSE, 
                            trim_ws = TRUE)

    names(r_commits) <- c("name", "date", "message")
    readr::write_csv(r_commits, 'r-commits.csv')
    ```

    Creates a `csv` file that `polars` can read.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The R Commits dataset""")
    return


@app.cell(hide_code=True)
def _(mo):
    import polars as pl

    df = pl.read_csv(
        mo.notebook_location() / "public" / "rcommits.csv",
        try_parse_dates=True,
    )

    df.head()

    return df, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    OK. The data is in, and as you can see it's a 3 column format with `name`, `date`, and (commit) `message`.

    What is fascinating is that there are only 34 unique contributors. This is totally unexpected for me. For such a big and old project, I was expecting hundreds of contributors.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## The contributors table""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Upon closer inspection, it seems that there are several contributors that are not persons. I am thinking maybe `apache` is for Apache Software Foundation, `(no author)` is well - when the author is not known(?), `root` and `r` may be system users or something used in automation.""")
    return


@app.cell(hide_code=True)
def _(df, pl):
    unique_names = df.select(pl.col("name").unique())

    unique_names
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next, I will exclude these four contributors and create a summary table that groups commit per year and contributor. And then make the data to a wide format in order to have the years as columns.""")
    return


@app.cell(hide_code=True)
def _(df, pl):
    exclude = ["apache", "(no author)", "r", "root"]

    df_summary = (
        df.filter(~pl.col("name").is_in(exclude))
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by(["year", "name"])
        .len(name="n_commits")
        .sort(["year", "n_commits"], descending=[True, True])
    )

    df_summary.head()
    return (df_summary,)


@app.cell
def _(df_summary):
    df_wide = df_summary.pivot(values="n_commits", index="name", on="year").fill_null(0)
    # Get all year columns as integers and sort ascending
    year_cols = sorted([c for c in df_wide.columns if c != "name"], key=int)

    # Select name first, then years in ascending order
    df_wide = df_wide.select(["name"] + year_cols)

    return df_wide, year_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The table 

    The idea is to have a **GitHub** like heatmap of contributions over the years by each contributor. In order to do this, a column that will hold the heatmap-like data in HTML need to be created. 

    There were two complications here: one to figure out where to start the `greens`, how light should the lightest green be; two creating a 'fake row' for the years in short format ('99 instead of 1999).
    """
    )
    return


@app.cell(hide_code=True)
def _(df_wide, pl, year_cols):
    import matplotlib
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import pandas as pd

    # Max value for color scaling
    values = df_wide.select(pl.all().exclude("name")).to_numpy().flatten()
    norm = mcolors.Normalize(vmin=1, vmax=max(values.max(), 1))
    greens = matplotlib.colormaps["Greens"]
    cmap = LinearSegmentedColormap.from_list(
        "truncated_greens",
        # start at 30% into Greens to have darker colors
        greens(np.linspace(0.3, 1, 256)),
    )

    # Function to build heatmap row (colored squares)
    def make_heatmap(row):
        values = row[1:]  # skip name
        cells = []
        for v in values:
            if v == 0:
                hexcolor = "#ffffff"  # white for zero commits
            else:
                rgba = cmap(norm(v))
                hexcolor = mcolors.to_hex(rgba)
            cells.append(
                f"<span title='{v} commits' "
                f"style='display:inline-block;width:16px;height:16px;"
                f"margin:1px;background:{hexcolor};border:1px solid #eee'></span>"
            )
        return "".join(cells)

    # Convert to pandas for row-wise apply
    df_heatmap = df_wide.to_pandas()
    df_heatmap["heatmap"] = df_heatmap.apply(make_heatmap, axis=1)

    # Create the row with years
    labels = []
    for i, y in enumerate(year_cols):
        if i == 0 or (i % 4 == 0) or i == len(year_cols):
            labels.append("'" + y[-2:])  # '1997' to '97'
        else:
            labels.append("&nbsp;")

    timeline = " ".join(
        f"<span style='display:inline-block;width:14px;text-align:center'>{year}</span>"
        for year in labels
    )

    # Add timeline as a "fake row" at the top
    timeline_row = {"name": "Year", "heatmap": timeline}
    df_final = df_heatmap[["name", "heatmap"]]
    df_final = pd.concat([pd.DataFrame([timeline_row]), df_final], ignore_index=True)

    return cmap, df_heatmap, matplotlib, mcolors, np


@app.cell
def _(mo):
    mo.md(r"""## The Great Table""")
    return


@app.cell(hide_code=True)
def _(df_heatmap, matplotlib, mcolors):
    import great_tables as gt
    from great_tables import html

    def make_source_and_legend(cmap):
        boxes = "".join(
            f"<span style='display:inline-block;width:16px;height:16px;"
            f"margin-left:2px;background:{mcolors.to_hex(cmap(i / 4))}'></span>"
            for i in range(5)
        )
        return html(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
            <div>Source: <a href="https://github.com/wch/r-source">Github</a> <a href="https://github.com/wch/r-source/commit/aa2e615f1f80ab536606c4c3db349fc9c372f479">#aa2e615</a></div>
            <div>less {boxes} more</div>
        </div>
        """)

    # Create the gt table
    tbl = (
        gt.GT(df_heatmap[["name", "heatmap"]])
        .tab_header(
            title=html(
                "<img src='https://www.r-project.org/Rlogo.png' style='height:32px; vertical-align:middle; margin-right:8px;'> R-Project Contributor Activity"
            ),
            subtitle="1997–2025, Commit counts by contributor and year",
        )
        .tab_options(column_labels_hidden=True)
        .tab_source_note(make_source_and_legend(matplotlib.colormaps["Greens"]))
        .tab_options(table_background_color="#F5F5F5")
        .opt_vertical_padding(scale=0.3)
    )

    tbl

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plotting with plotnine

    I found plotting with `plotnine` to be more or less the same as plotting with `ggplot2` which is kind of nice on its own. However, when I wanted to make the plot a somewhat interactive, I could not figure a way to do it (I was looking to add a filter per contributor and have the plot change by selecting different names.)

    It seems that that type of thing is possible to do in `marimo` with the python library [`altair`](https://altair-viz.github.io/), which as far as I understood is based on a grammar for graphics implementation just like `plotnine` and `ggplot2`. So that it something to maybe test in the future.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The plotnine plot

    The plot shows the number of commits for the whole repository over the period that the data is available for (1997-2025) with the size of the bubbles equaling the average word count for the commit messages in a given year. 

    What can this plot tell us? Could be that number of commits gets lower as the project matures, but commits require longer explanations. This could be a hypothesis to test. 
    """
    )
    return


@app.cell(hide_code=True)
def _(cmap, df, matplotlib, np, pl):
    import plotnine as p9

    df_plot = df.with_columns(
        [
            df["date"].dt.year().alias("year"),
            df["message"].str.split(" ").list.len().alias("word_count"),
        ]
    )

    # Summarise: commits per year and average word_count
    df_summary_plot = (
        df_plot.group_by("year")
        .agg(
            [
                pl.len().alias("n_commits"),
                pl.mean("word_count").round(0).alias("avg_words"),
            ]
        )
        .sort("year")
    )

    # Convert to pandas for plotnine
    pdf = df_summary_plot.to_pandas()

    vlines = [2000, 2004, 2013, 2020]
    vlabels_position = [2001.0, 2005.0, 2014.0, 2021.0]
    vlabels = ["R 1.0", "R 2.0", "R 3.0", "R 4.0"]
    colors = [matplotlib.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)]

    # Bubble plot for average words per commit over years
    p = (
        p9.ggplot(
            pdf, p9.aes(x="year", y="n_commits", size="avg_words", fill="avg_words")
        )
        + p9.geom_point(
            shape="o", alpha=0.8, color="black"
        )  # shape="o" uses fill + border
        + p9.geom_vline(xintercept=vlines, linetype="dashed", color="gray")
        + p9.annotate(
            "text",
            x=vlabels_position,
            y=1,
            label=vlabels,
            angle=0,
            va="bottom",
            ha="center",
            size=8,
        )
        + p9.guides(
            fill=p9.guide_legend("Avg. Words per Commit"),
            size=p9.guide_legend("Avg. Words per Commit"),
        )
        + p9.scale_size_continuous(range=(5, 15))
        + p9.scale_fill_gradientn(colors=colors)
        + p9.labs(
            x="Year",
            y="Number of Commits",
            title="R-Project Commit Activity Over Time",
            caption="Source: GitHub #aa2e615",
        )
        + p9.theme_minimal(base_size=14)
        + p9.theme(
            plot_background=p9.element_rect(fill="#F5F5F5", color="#F5F5F5"),
            panel_background=p9.element_rect(fill="#F5F5F5", color="#F5F5F5"),
            plot_title=p9.element_text(
                weight="bold", size=18, margin={"b": 15, "t": 15}
            ),
            figure_size=(12, 8),
        )
        + p9.theme(legend_position="bottom")
    )

    p.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Maybe next steps

    I can imagine this whole notebook to be automated using `GitHub actions` in order to track commits over time. 

    Maybe a nice idea would be to have a word cloud of the commit messages.
    """
    )
    return


if __name__ == "__main__":
    app.run()
