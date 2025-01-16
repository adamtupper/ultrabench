import os


def save_version_info(output_dir, version):
    """Create a file with the UltraBench version and current date."""
    with open(os.path.join(output_dir, "version.txt"), "w") as f:
        f.write(f"UltraBench Version: {version}\n")
        f.write(f"Created: {os.popen('date').read().strip()}\n")
