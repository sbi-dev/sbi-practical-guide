from invoke import task
from pathlib import Path

basepath = "/Users/michaeldeistler/Documents/phd/tutorial_paper"

open_cmd = "open"

fig_names = {
    "1": "paper/fig1_introduction",
    "2": "paper/fig2_ball_throw",
    "3": "paper/fig3_workflow",
    "4": "paper/fig4_sbc",
    "5": "paper/fig5_sbi_methods",
    "6": "paper/fig6_grav_wave",
    "7": "paper/fig7_drift_diffusion",
    "8": "paper/fig8_pyloric",
    "9": "paper/fig_3_2_3_prior_pitfalls/res/two_moons",
}

@task
def convertpngpdf(c, fig):
    _convertsvg2pdf(c, fig)
    _convertpdf2png(c, fig)


########################################################################################
# Helpers
########################################################################################
@task
def _convertsvg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.svg")
    for path in pathlist:
        c.run(f"inkscape {str(path)} --export-pdf={str(path)[:-4]}.pdf")


@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.pdf")
    for path in pathlist:
        c.run(
            f'inkscape {str(path)} --export-png={str(path)[:-4]}.png -b "white" --export-dpi=450'
        )
