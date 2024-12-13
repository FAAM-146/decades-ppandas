import os
import sys
import requests
import subprocess

import ppodd


class TagError(Exception):
    def __init__(self, message: str, tag: str, giterr: str) -> None:
        super().__init__(message)
        self.tag = tag
        self.giterr = giterr


class GitHubError(Exception):
    def __init__(self, message: str, giterr: str) -> None:
        super().__init__(message)
        self.giterr = giterr


def get_token(env_var: str, filename: str) -> str:
    """
    Get a token from an environment variable or a file in the user's home directory.

    Args:
        env_var (str): The name of the environment variable to check for the token.
        filename (str): The name of the file to check for the token.

    Returns:
        str: The token
    """
    if env_var in os.environ:
        return os.environ[env_var]

    home_dir = os.path.expanduser("~")
    token_file = os.path.join(home_dir, filename)
    if os.path.exists(token_file):
        with open(token_file) as f:
            return f.read().strip()

    raise RuntimeError(f"No token found in {env_var} or {token_file}")


def get_zenodo_token() -> str:
    """
    Get the Zenodo token from the environment or a file in the user's home directory.

    Returns:
        str: The Zenodo token
    """
    return get_token("ZEONDO_TOKEN", ".zenodo")


def get_github_token() -> str:
    """
    Get the GitHub token from the environment or a file in the user's home directory.

    Returns:
        str: The GitHub token
    """
    return get_token("GITHUB_TOKEN", ".github")


def tag_release() -> str:
    """
    Tag the current release in git, using the current version number and
    push the tag to the remote.

    Returns:
        str: The version number
    """
    version = f"v{ppodd.version()}"
    try:
        subprocess.run(["git", "tag", version], check=True, capture_output=True)
        print(f"Tagged release {version}")
    except subprocess.CalledProcessError as e:
        raise TagError(
            f"Error creating tag", tag=version, giterr=e.stderr.decode().strip()
        ) from e

    try:
        subprocess.run(
            ["git", "push", "origin", version], check=True, capture_output=True
        )
        print(f"Pushed tag {version} to remote")
    except subprocess.CalledProcessError as e:
        raise GitHubError(
            f"Failed to push tag {version} to remote", giterr=e.stderr.decode.strip()
        ) from e

    return version

def get_patch_version() -> str:
    """
    Get the patch version of the current release.

    Returns:
        str: The patch version
    """
    return ppodd.version().split(".")[2]


def create_git_release(description: str) -> str:
    """
    Create a release on GitHub using the current version number.

    Returns:
        str: The version number
    """
    version = f"v{ppodd.version()}"
    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = {
        "tag_name": version,
        "name": version,
        "target_commitish": "master",
        "body": description,
        "draft": False,
        "prerelease": False,
    }
    r = requests.post(
        f"https://api.github.com/repos/{ppodd.GITHUB_ORG}/{ppodd.GITHUB_REPO}/releases",
        headers=headers,
        json=data,
    )

    if r.status_code != 201:
        raise RuntimeError(f"Failed to create release: {r.text}")

    return version


def main():
    zenodo_token = get_zenodo_token()
    # github_token = get_github_token()

    print()
    print("=" * 50)
    print("Creating new decades-ppandas (ppodd) release")
    print("=" * 50, "\n")

    repo_directory = os.path.abspath(os.path.join(os.path.dirname(ppodd.__file__), ".."))
    if os.getcwd() != repo_directory:
        print(f"Note: Moving into repository: {repo_directory}\n")
        os.chdir(repo_directory)

    print("Has this release been checked against existing data? [y/n]")
    checked = input("> ")
    if checked.lower() != "y":
        print("Please check the data before proceeding.\n")
        sys.exit(1)

    print(f"This release will be tagged as v{ppodd.version()}. Continue? [y/n]")
    proceed = input("> ")
    if proceed.lower() != "y":
        print("Exiting without creating a release.\n")
        sys.exit(1)

    if get_patch_version() != "0":
        print("Patch release. Force generation of documentation? [y/n]")
        force_str = input("> ")
        gen_docs = force_str.lower() == "y"
    else:
        gen_docs = True

    try:
        release_version = tag_release()
    except TagError as e:
        print(f"Could not create tag {e.tag}. Git error: {e.giterr}\n")
        # sys.exit(1)
    except GitHubError:
        print("Failed to push tag to remote. You may need to do this manually.\n")
        sys.exit(1)

    if gen_docs:
        print("Generating documentation [web]... ")
        with ppodd.flipdir("ppodd/docs/data"):
            subprocess.run(["make", "html"], check=True)
            
        print("Generating documentation [pdf]... ")
        with ppodd.flipdir("ppodd/docs/data"):
            subprocess.run(["make", "latexpdf"], check=True)

if __name__ == "__main__":
    main()
