#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/plot_results.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: September 23, 2026
# PURPOSE: Charts for comparing the models, read from the saved output files:
#            plot_accuracy()        - each accuracy measure, one panel each
#            plot_speed_vs_accuracy() - seconds per image against breed accuracy
#            plot_mistakes()        - the images a model got wrong, with its
#                                     top guesses
#          Used by notebooks/run_on_colab.ipynb; needs matplotlib.
##
import csv
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from print_model_tables import PET_IMAGE_FILES, read_results

# One series colour (all bars/points are the same kind of thing; the model
# name is on the axis or next to the point), on a light chart surface.
# Text uses ink colours, never the series colour.
SERIES = '#2a78d6'
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'

# Panels for plot_accuracy(), left to right: (statistic key, title).
ACCURACY_PANELS = (('pct_correct_dogs', 'Dogs identified as dogs'),
                   ('pct_correct_notdogs', 'Non-dogs identified as non-dogs'),
                   ('pct_correct_breed', 'Dog breeds named correctly'),
                   ('pct_match', 'Labels matched'))


def _percent_label(value):
    """'100%', '82.5%', '93.3%': one decimal, dropped when it's zero."""
    return '{:.1f}'.format(value).rstrip('0').rstrip('.') + '%'


def _style(ax):
    """Recessive axes: hairline solid gridlines, muted ticks, no box."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, length=0)
    ax.grid(color=GRID, linewidth=1, linestyle='-')
    ax.set_axisbelow(True)


def load_results(files=PET_IMAGE_FILES):
    """Returns [(model, stats)] for the usable output files, in order."""
    return read_results(files)


def plot_accuracy(files=PET_IMAGE_FILES, title='Accuracy by model'):
    """
    One panel per accuracy measure, one horizontal bar per model, with the
    value at the bar tip. Returns the matplotlib Figure.
    """
    results = load_results(files)
    models = [model for model, _ in results]
    fig, axes = plt.subplots(1, len(ACCURACY_PANELS), sharey=True,
                             figsize=(3.2 * len(ACCURACY_PANELS), 0.4 * len(models) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, panel_title) in zip(axes, ACCURACY_PANELS):
        values = [stats.get(key, 0.0) for _, stats in results]
        positions = range(len(models))
        # Thin bars (well under half the band), from a shared 0 baseline.
        ax.barh(positions, values, height=0.4, color=SERIES)
        for y, value in zip(positions, values):
            ax.text(value + 2, y, _percent_label(value), va='center',
                    color=TEXT_SECONDARY, fontsize=9)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(['0%', '50%', '100%'])
        ax.grid(axis='y', visible=False)
        ax.set_title(panel_title, color=TEXT, fontsize=10, loc='left')
        _style(ax)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    fig.suptitle(title, color=TEXT, x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    return fig


def plot_speed_vs_accuracy(files=PET_IMAGE_FILES,
                           title='Speed vs breed accuracy (up and left is better)'):
    """
    One labelled point per model: seconds per image (x) against the share of
    dog breeds named correctly (y). Models without a measured speed are left
    out. Returns the matplotlib Figure.
    """
    results = [(model, stats) for model, stats in load_results(files)
               if 'sec_per_image' in stats]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    points = sorted((stats['sec_per_image'], stats['pct_correct_breed'], model)
                    for model, stats in results)
    # >= 8px markers with a 2px surface ring, so overlapping points stay readable.
    ax.scatter([x for x, _, _ in points], [y for _, y, _ in points], s=90, color=SERIES,
               edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set_xlim(0, max([x for x, _, _ in points] + [0.01]) * 1.3)
    ax.set_ylim(-4, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    fig.canvas.draw()  # so label sizes can be measured
    _label_points(fig, ax, points)
    ax.set_xlabel('Seconds per image', color=TEXT_SECONDARY)
    ax.set_ylabel('Dog breeds named correctly (%)', color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT, fontsize=12, loc='left')
    if not results:
        ax.text(0.5, 0.5, 'No measured speeds yet: run the batch scripts',
                transform=ax.transAxes, ha='center', color=MUTED)
    fig.tight_layout()
    return fig


def _label_points(fig, ax, points):
    """
    Labels each (x, y, name) point without overlapping another label or a
    point. Tries right, left, above and below the point; if all are taken,
    moves the label further out and joins it to its point with a thin line.
    """
    renderer = fig.canvas.get_renderer()
    marker_radius = 7  # pixels: the 90pt^2 marker plus its ring
    points_px = [ax.transData.transform((x, y)) for x, y, _ in points]
    taken = [(px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius)
             for px, py in points_px]

    def overlaps(box):
        return any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                   for b in taken)

    # (x offset, y offset, horizontal alignment, vertical alignment) in points.
    near = [(9, 0, 'left', 'center'), (-9, 0, 'right', 'center'),
            (0, 9, 'center', 'bottom'), (0, -9, 'center', 'top')]
    far = [(dx * step, dy * step, ha, va) for step in (2, 3, 4)
           for dx, dy, ha, va in ((9, 9, 'left', 'bottom'), (9, -9, 'left', 'top'),
                                  (-9, 9, 'right', 'bottom'), (-9, -9, 'right', 'top'))]
    for (x, y, name), (px, py) in zip(points, points_px):
        for i, (dx, dy, ha, va) in enumerate(near + far):
            label = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords='offset points',
                                ha=ha, va=va, color=TEXT_SECONDARY, fontsize=9)
            extent = label.get_window_extent(renderer)
            box = (extent.x0 - 2, extent.y0 - 2, extent.x1 + 2, extent.y1 + 2)
            if not overlaps(box) or i == len(near + far) - 1:
                break
            label.remove()
        if i >= len(near):
            # Placed away from its point: draw a leader line back to it.
            ax.annotate('', (x, y), xytext=(dx, dy), textcoords='offset points',
                        arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8,
                                        shrinkA=0, shrinkB=5))
        taken.append(box)


def plot_mistakes(csv_path, image_dir, max_images=12, columns=4):
    """
    Shows the images a model got wrong (labels don't match), each captioned
    with the real label and the model's top guesses. Reads a CSV written by
    check_images.py --csv. Returns the matplotlib Figure, or None if the
    model got every image right.
    """
    with open(csv_path, newline='') as infile:
        wrong = [row for row in csv.DictReader(infile) if row['labels_match'] == '0']
    if not wrong:
        return None
    wrong = wrong[:max_images]
    rows = (len(wrong) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.6 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, wrong):
        with Image.open(os.path.join(image_dir, row['filename'])) as img:
            ax.imshow(ImageOps.exif_transpose(img).convert('RGB'))
        # Same-sized cells whatever the image shape, so the captions line up.
        ax.set_aspect('equal', adjustable='datalim')
        guesses = row['top_guesses'].split('; ')[:3]
        # Keep only the first name of each guess, e.g. "walker hound (55.0%)".
        short = ['{} ({}'.format(g.split(' (')[0].split(',')[0], g.rsplit(' (', 1)[-1])
                 for g in guesses]
        ax.set_title('real: {}\n'.format(row['pet_label']) + '\n'.join(short),
                     color=TEXT, fontsize=9, loc='left')
    model = wrong[0]['model']
    fig.suptitle('Images {} got wrong, with its top guesses'.format(model),
                 color=TEXT, x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    return fig
