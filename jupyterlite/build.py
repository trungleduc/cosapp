from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).parent
ROOT = HERE.parent

CONTENTS_DIR = HERE / "contents"
NB_SOURCE = ROOT / "docs" / "tutorials"
LITE_DIR = HERE / "dist"
IGNORE_LIST = ["FMI.ipynb", "exprampode_fmu.py"]


def copy_cosapp_package():
    """
    Copies the `cosapp` package tarball from the distribution directory to the
    current directory.

    Raises:
        FileNotFoundError: If the `cosapp` package tarball is not found in the
        `dist` directory.
    """

    dist_dir = ROOT / "dist"
    if not dist_dir.exists():
        raise FileNotFoundError("Missing cosapp package")
    dest = HERE / "cosapp.tar.gz"
    for f in dist_dir.glob("*tar.gz"):
        shutil.copyfile(f, dest)
        break
    else:
        raise FileNotFoundError("Missing cosapp package")


def copy_example_notebooks():
    """
    Copies example Jupyter notebooks from the tutorial to the contents
    directory, excluding files listed in IGNORE_LIST.

    Raises:
        FileNotFoundError: If the source directory containing the notebooks
        is not found.
    """

    if not NB_SOURCE.exists():
        raise FileNotFoundError("Missing notebooks directory")

    if CONTENTS_DIR.exists():
        shutil.rmtree(CONTENTS_DIR)
    shutil.copytree(NB_SOURCE, CONTENTS_DIR)

    for f in CONTENTS_DIR.glob("*"):
        if f.name in IGNORE_LIST:
            f.unlink()


def build_lite_site():
    """
    Builds the JupyterLite site by copying necessary packages and example
    notebooks, and then running the JupyterLite build command.
    
    Raises:
        subprocess.CalledProcessError: If the JupyterLite build command fails.
    """

    copy_cosapp_package()
    copy_example_notebooks()

    if LITE_DIR.exists():
        shutil.rmtree(LITE_DIR)

    cmd = [
        "jupyter",
        "lite",
        "build",
        "--contents",
        CONTENTS_DIR,
        "--output-dir",
        LITE_DIR,
        "--XeusAddon.mount_jupyterlite_content=True",
        f"--XeusAddon.mounts={ROOT}/cosapp:/lib/python3.11/site-packages/cosapp",
        f"--XeusAddon.mounts={HERE}/.cosapp.d:/home/web_user/.cosapp.d",
    ]

    subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    build_lite_site()
