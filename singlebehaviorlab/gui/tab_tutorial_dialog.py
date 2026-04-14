"""
Per-tab help popups with detailed NOR-themed tutorials.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPushButton, QTextBrowser, QVBoxLayout


def _wrap_html(title: str, body: str) -> str:
    return (
        f"<h2>{title}</h2>"
        "<p><b>Example: Novel Object Recognition (NOR)</b><br>"
        "Overhead video of a mouse in an arena with one familiar object and one novel object. "
        "Typical behaviors: <i>sniff_novel</i>, <i>sniff_familiar</i>, <i>walk_arena</i>, "
        "<i>rear</i>, <i>groom</i>, <i>freeze</i>. Adapt names to your own protocol.</p>"
        f"{body}"
    )


TAB_TUTORIALS: dict[str, dict[str, str]] = {
    "labeling": {
        "title": "Labeling",
        "body": _wrap_html(
            "Labeling",
            """
            <h3>What this tab is for</h3>
            <p>Start here when you have raw NOR videos and want to build the first labeled training set. 
            This tab extracts short clips, lets you define behavior classes, and lets you label either whole clips or frame ranges inside clips.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Add one or more source videos from your NOR sessions.</li>
              <li>Set <b>Target FPS</b> and <b>Frames per clip</b>. A good first setting is <b>12 fps</b> and <b>8 frames</b> per clip.</li>
              <li>Set <b>Step frames</b> to control how densely clips are extracted. Use a smaller step for denser coverage.</li>
              <li>If the videos are very long, set <b>Max clips per video</b> so the first pass stays manageable.</li>
              <li>Click <b>Extract all clips</b>. Clips are saved into <code>data/clips/</code>.</li>
              <li>Create behavior classes. For NOR, common classes are <i>sniff_novel</i>, <i>sniff_familiar</i>, <i>walk_arena</i>, <i>groom</i>, and <i>rear</i>.</li>
              <li>Start by collecting about <b>10–20 clean clips per class</b>. That is enough for a first training round, and you can always add more later in the <b>Refine</b> stage.</li>
              <li>Use the <b>Random</b> button often instead of labeling clips strictly in sequence. This usually gives a more diverse first set across videos, animals, and contexts.</li>
              <li>Select a clip and assign a class with the class buttons or keys <b>1–9</b>.</li>
              <li>At the beginning, focus on <b>clean 8-frame clips</b> where the behavior is isolated and obvious. You do <b>not</b> need to label transition clips first as long as you have good isolated examples of the behavior of interest.</li>
              <li>Use <b>per-frame labeling</b> later when transitions are important for improving boundaries, such as walking changing into sniff_novel within one clip.</li>
              <li>Press <b>Ctrl+S</b> to save.</li>
            </ol>

            <h3>Helpful optional tools</h3>
            <ul>
              <li><b>Show unlabeled only</b> and <b>Next unlabeled</b> help you move through the dataset quickly.</li>
              <li><b>Fullscreen</b> helps when the nose-object interaction is subtle.</li>
              <li><b>Multi-label</b> is useful if you want OvR training later.</li>
              <li><b>Hard-negative round dataset</b> is helpful when two classes look similar, such as sniff_familiar vs general head-down exploration.</li>
            </ul>

            <h3>Localization bbox</h3>
            <p>VideoPrism sees clips at <b>288×288</b>. If the mouse is small in the frame because the arena is large or the camera is far away, 
            draw a localization box around the animal and save it. This helps the model crop in before classification. 
            In many normal NOR setups, you can try training without this first.</p>
            """,
        ),
        "next_tab": "Training Sequencing Model",
        "next_hint": "Once you have enough labeled clips across your behaviors, train the model.",
    },
    "training": {
        "title": "Training Sequencing Model",
        "body": _wrap_html(
            "Training Sequencing Model",
            """
            <h3>What this tab is for</h3>
            <p>This is where labeled clips become a classifier. The model learns how to separate the NOR behaviors you defined in Labeling.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Check that the <b>annotation file</b> and <b>clips directory</b> point to your experiment.</li>
              <li>Use <b>Dataset info</b> to confirm that class counts look reasonable.</li>
              <li>Start with a preset profile such as <b>LowInputData</b> if you are unsure.</li>
              <li>Choose epochs, batch size, and learning rates if you want to tune manually.</li>
              <li>Leave the <b>temporal decoder</b> on unless your dataset is extremely small. It helps sequence frame-level context inside the clip.</li>
              <li>For modest datasets, turn on <b>data augmentation</b>. This is usually important, not just optional, because it improves generalization across sessions, lighting, and animal-to-animal variation.</li>
              <li>Decide early whether to use <b>OvR</b> or standard softmax. OvR is an important modeling choice when classes can overlap or when you want one binary head per behavior.</li>
              <li>Use a <b>validation split</b> when possible so you can compare checkpoints more reliably.</li>
              <li>Click <b>Visualize training</b> if you want live loss and F1 plots.</li>
              <li>Click <b>Start training</b>. The best checkpoint is saved to <code>models/behavior_heads/</code>.</li>
            </ol>

            <h3>Additional training tools</h3>
            <ul>
              <li><b>Confusion-aware hard mining</b> focuses training on clips the model gets wrong.</li>
              <li><b>Fine-tune from pretrained</b> is useful if you already trained on a similar experiment and only want to adapt.</li>
              <li><b>Auto-tune</b> can try multiple settings automatically before a final run.</li>
              <li><b>Batch Train</b> is useful if you want to compare several profiles in one go.</li>
            </ul>

            <h3>Localization</h3>
            <p>The Localization section only becomes active when bbox labels exist. If your NOR animal is small relative to the arena, 
            training can first learn localization and then classify on the crop.</p>
            """,
        ),
        "next_tab": "Sequencing",
        "next_hint": "Load the trained checkpoint and run it on new NOR test videos.",
    },
    "sequencing": {
        "title": "Sequencing",
        "body": _wrap_html(
            "Sequencing",
            """
            <h3>What this tab is for</h3>
            <p>This tab runs your trained model on unseen videos and converts predictions into a behavior timeline.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Load the trained <code>.pt</code> checkpoint.</li>
              <li>Select one or more NOR videos to score.</li>
              <li>Match clip settings to training: <b>FPS</b>, <b>clip length</b>, <b>step</b>, and <b>resolution</b>.</li>
              <li>Run inference.</li>
              <li>Inspect the resulting timeline. For NOR, verify that object investigation periods are where you expect them.</li>
            </ol>

            <h3>Helpful optional tools</h3>
            <ul>
              <li><b>Quick-check sampled inference</b> runs only a few chunks first, so you can verify behavior quality before processing a whole session.</li>
              <li><b>Viterbi smoothing</b> and segment merging reduce flicker between nearby labels.</li>
              <li><b>Collect attention maps</b> helps you verify whether the model is looking at the mouse-object interaction area.</li>
              <li>Click on timeline segments to open a clip popup with scores and localization overlays.</li>
            </ul>

            <h3>Exports</h3>
            <p>You can export JSON results, CSV/SVG timelines, and overlay videos. For NOR, overlay videos are often the fastest sanity check before formal analysis.</p>
            """,
        ),
        "next_tab": "Refine",
        "next_hint": "Review uncertain or boundary clips and feed corrections back into training.",
    },
    "refine": {
        "title": "Refine",
        "body": _wrap_html(
            "Refine",
            """
            <h3>What this tab is for</h3>
            <p>Refine is the active-learning loop. Instead of relabeling everything, you focus on the clips that matter most for improving the model.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Load the uncertainty or inference results if they are not already passed in automatically.</li>
              <li>Pick a review mode:
                <ul>
                  <li><b>Uncertain</b> — best for fixing model mistakes.</li>
                  <li><b>Confident</b> — best for adding more easy training data fast.</li>
                  <li><b>Transition</b> — best for improving behavior boundaries.</li>
                </ul>
              </li>
              <li>Inspect each suggested clip, especially cases where the mouse briefly inspects an object and then walks away.</li>
              <li>Accept, relabel, or mark as hard negative.</li>
              <li>Save the updated clips back into annotations.</li>
            </ol>

            <h3>When to use it in NOR</h3>
            <p>This is especially useful when <i>sniff_novel</i> and <i>sniff_familiar</i> are confused, or when very short transitions between walking and investigation create noisy boundaries.</p>
            """,
        ),
        "next_tab": "Training Sequencing Model",
        "next_hint": "Retrain with the refined clips, then run Sequencing again on held-out videos.",
    },
    "analysis": {
        "title": "Downstream Analysis",
        "body": _wrap_html(
            "Downstream Analysis",
            """
            <h3>What this tab is for</h3>
            <p>This is where predictions become summary figures, spatial behavior maps, and statistical comparisons.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Load inference or bout-level outputs from your experiment.</li>
              <li>Use the overview plots to summarize <b>occurrences</b>, <b>average bout duration</b>, <b>total duration</b>, and <b>percent time</b>.</li>
              <li>For NOR, compare <b>time on novel</b> vs <b>time on familiar</b>, or compare groups such as control vs treated.</li>
              <li>Use <b>spatial distribution</b> if you want to see where behaviors happen in the arena.</li>
              <li>Use the <b>transition graph</b> if you want to understand sequences such as walk_arena → sniff_novel → groom.</li>
            </ol>

            <h3>Helpful optional tools</h3>
            <ul>
              <li><b>Group comparison</b> is useful when you have multiple animals or conditions.</li>
              <li>Statistical tests include Mann-Whitney and Kruskal-Wallis depending on how many groups you compare.</li>
              <li>You can export figures as PDF, SVG, PNG, or HTML, and tables as CSV.</li>
            </ul>
            """,
        ),
        "next_tab": "Labeling or Segmentation Tracking",
        "next_hint": "Either continue your supervised loop with new labels, or start the unbiased discovery path on a larger cohort.",
    },
    "segmentation": {
        "title": "Segmentation Tracking",
        "body": _wrap_html(
            "Segmentation Tracking",
            """
            <h3>What this tab is for</h3>
            <p>This is the first stage of the unbiased discovery path. It isolates animals from many videos so later embeddings reflect the animal's behavior rather than the full frame layout.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Load one or more NOR videos.</li>
              <li>Choose a SAM2 model and tracking resolution.</li>
              <li>On a clear frame, click <b>positive</b> points on the mouse and <b>negative</b> points on background, shadows, or objects when needed.</li>
              <li>Use object IDs if more than one animal must be tracked.</li>
              <li>Run tracking.</li>
              <li>If the mask drifts, pause, add corrective points, and resume.</li>
              <li>Save masks and optionally an overlay video.</li>
            </ol>

            <h3>Why this matters for discovery</h3>
            <p>When you want unbiased differences across many animals and sessions, standardized animal isolation reduces variation caused by arena framing and camera placement.</p>
            """,
        ),
        "next_tab": "Registration",
        "next_hint": "Use the saved masks to build consistent animal-centered crops and extract embeddings.",
    },
    "registration": {
        "title": "Registration",
        "body": _wrap_html(
            "Registration",
            """
            <h3>What this tab is for</h3>
            <p>Registration converts segmented videos into normalized, animal-centered clips and then extracts VideoPrism embeddings.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Load video and mask pairs.</li>
              <li>Choose crop box size, target size, normalization, and whether ROI stays locked or updates frame-by-frame.</li>
              <li>Set clip extraction settings. For consistency with Labeling, 12 fps and 8 frames are a good first pass.</li>
              <li>Process videos into registered clips.</li>
              <li>Click <b>Extract embeddings</b> to run VideoPrism and save the matrix plus metadata.</li>
            </ol>

            <h3>Why this matters for discovery</h3>
            <p>This step makes later clustering less biased by background or scale, so clusters are more likely to reflect behavior rather than nuisance differences.</p>
            """,
        ),
        "next_tab": "Clustering",
        "next_hint": "Use embeddings to reveal behavior structure without predefining labels.",
    },
    "clustering": {
        "title": "Clustering",
        "body": _wrap_html(
            "Clustering",
            """
            <h3>What this tab is for</h3>
            <p>Clustering is the heart of the unbiased discovery path. Instead of deciding the important behaviors first, you let the embedding space reveal repeated patterns across many sessions and animals.</p>

            <h3>How to use it</h3>
            <ol>
              <li>Load the embedding matrix and metadata from Registration.</li>
              <li>Optionally preprocess with scaling or PCA.</li>
              <li>Run UMAP to obtain a lower-dimensional view.</li>
              <li>Run Leiden or HDBSCAN to form clusters.</li>
              <li>Inspect representative points from each cluster.</li>
              <li>Click directly on points in the embedding plot to open the corresponding clips and see what kind of behavior repeatedly appears in that region of the cluster.</li>
              <li>For NOR, clusters may reveal repeated modes such as object approach, nose contact, locomotion around the perimeter, grooming away from the object, or rearing.</li>
            </ol>

            <h3>How to proceed</h3>
            <p>After you discover stable clusters, switch to <b>Labeling</b> and use the setup dialog to load representative clips. Use the clicked example clips to judge what behavior is most common in that cluster, then assign names such as <i>sniff_novel</i> or <i>walk_arena</i> and move into supervised training.</p>
            """,
        ),
        "next_tab": "Labeling",
        "next_hint": "Name discovered clusters and turn them into a diverse labeled training set.",
    },
}


def show_tab_tutorial(parent, tab_id: str) -> None:
    data = TAB_TUTORIALS.get(tab_id)
    if not data:
        return

    dlg = QDialog(parent)
    dlg.setWindowTitle(f"{data['title']} Guide")
    dlg.setMinimumSize(720, 600)
    layout = QVBoxLayout(dlg)

    browser = QTextBrowser()
    browser.setOpenExternalLinks(False)
    browser.setHtml(data["body"])
    layout.addWidget(browser, 1)

    next_lbl = QLabel(
        f"<b>Next:</b> open tab <i>{data['next_tab']}</i> — {data['next_hint']}"
    )
    next_lbl.setWordWrap(True)
    next_lbl.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(next_lbl)

    next_title = data["next_tab"]
    if hasattr(parent, "tabs") and " or " not in next_title and next_title:
        go_btn = QPushButton("Open next tab")
        go_btn.setToolTip(f"Switch to: {next_title}")

        def _switch_next() -> None:
            tw = parent.tabs
            for i in range(tw.count()):
                if tw.tabText(i) == next_title:
                    tw.setCurrentIndex(i)
                    dlg.accept()
                    return

        go_btn.clicked.connect(_switch_next)
        layout.addWidget(go_btn)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    dlg.exec()
