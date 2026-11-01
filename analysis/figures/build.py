"""Make every figure and the report from saved results."""
import traceback
from functools import lru_cache

from tqdm.auto import tqdm

from steinmetz import examples, report
from steinmetz.results import Results

from . import assignment, ext_figs, main_figs


def make_all(cfg, only=None):
    res = Results(cfg)
    chosen = lru_cache(maxsize=1)(lambda: examples.choose(res))  # needs regression results, so only when a figure asks
    jobs = {
        "fig1": lambda: main_figs.fig1(res), "fig2": lambda: main_figs.fig2(res, chosen()), "fig3": lambda: main_figs.fig3(res, chosen()),
        "fig4": lambda: main_figs.fig4(res, chosen()), "fig5": lambda: main_figs.fig5(res),
        "ext1": lambda: ext_figs.ext1(res), "ext2": lambda: ext_figs.ext2(res), "ext3": lambda: ext_figs.ext3(res), "ext4": lambda: ext_figs.ext4(res, chosen()),
        "ext5": lambda: ext_figs.ext5(res, chosen()), "ext6": lambda: ext_figs.ext6(res), "ext7": lambda: ext_figs.ext7(res), "ext8": lambda: ext_figs.ext8(res),
        "ext9": lambda: ext_figs.ext9(res, chosen()), "assignment": lambda: assignment.all_assignment(res, chosen()), "report": lambda: report.write(res),
    }
    todo = {k: v for k, v in jobs.items() if not only or k in only}
    failed = {}
    for name, fn in tqdm(todo.items(), desc="figures", unit="figure", disable=len(todo) < 2):
        try:
            fn()
        except Exception:  # one failed figure must not lose the others after a long run
            failed[name] = traceback.format_exc()
            tqdm.write(f"FAILED {name}\n{failed[name]}")
    return failed
