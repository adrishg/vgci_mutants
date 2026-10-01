"""Execute every code cell with rich outputs, without opening kernel TCP ports.

An in-process IPython shell is useful in restricted environments. Each notebook
runs in a fresh Python subprocess, with the same inline display backend as Jupyter.
Use normal Jupyter/nbclient execution interchangeably outside the sandbox.
"""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def execute(path):
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    shell = InteractiveShell.instance()
    os.chdir(path.parent)  # verifies the usual channel-directory launch
    nb = nbformat.read(path, as_version=4)
    count = 0
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'{path.name}: cell {index}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        cell.outputs = []
        for name, text in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if text:
                cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=text))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        nbformat.write(nb, path)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'{path}: cell {index} failed: {result.error_before_exec or result.error_in_exec}\n{captured.stdout}\n{captured.stderr}')
    print(f'Completed {path.name}: {count} code cells', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        execute(Path(sys.argv[1]).resolve())
    else:
        for path in sorted(ROOT.glob('*/*_MolProbity_clashscore_analysis.ipynb')):
            subprocess.run([sys.executable, __file__, str(path)], check=True)
