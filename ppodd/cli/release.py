import argparse
import json
import os
import sys
from typing import Any
import requests
import subprocess

import ppodd

TICK = "\033[92m✔\033[0m"
CROSS = "\033[91m✗\033[0m"

ZENODO_METADATA = {
    "metadata": {
        "title": "FAAM Core Data Product",
        "upload_type": "publication",
        "publication_type": "datapaper",
        "description": (
            "This document describes the processing involved in producing the FAAM core data product, and the structure of the resulting data."
        ),
        "creators": [
            {
                "name": "Dave Sproson",
                "affiliation": "FAAM Airborne Laboratory",
                "orcid": "0000-0001-6806-2599",
            }
        ],
        "version": ppodd.version(),
    }
}

ZENODO_CONCEPT_ID = {"sandbox": "143243", "live": "7105518"}


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
    return get_token("ZENODO_TOKEN", ".zenodo")


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
    except subprocess.CalledProcessError as e:
        raise TagError(
            f"Error creating tag", tag=version, giterr=e.stderr.decode().strip()
        ) from e

    try:
        subprocess.run(
            ["git", "push", "origin", version], check=True, capture_output=True
        )
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


def create_git_release(description: str, token: str) -> str:
    """
    Create a release on GitHub using the current version number.

    Returns:
        str: The version number
    """
    version = f"v{ppodd.version()}"
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


def zenodo_get_latest_version(
    token_str: str, concept_id: str, use_sandbox: bool
) -> str:
    """
    Get the latest version of a Zenodo release.

    Args:
        token_str (str): The Zenodo access token
        concept_id (str): The concept ID for the release
        use_sandbox (bool): Whether to use the Zenodo sandbox

    Returns:
        str: The latest version
    """
    params = {"access_token": token_str}
    prefix = "sandbox." if use_sandbox else ""
    r = requests.get(
        f"https://{prefix}zenodo.org/api/records/{concept_id}/versions/latest",
        params=params,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Failed to get latest version: {r.text}")

    return r.json()["id"]


def zenodo_create_new_version(
    token_str: str, concept_id: str, use_sandbox: bool
) -> dict[str, Any]:
    """
    Create a new version of a Zenodo release.

    Args:
        token_str (str): The Zenodo access token
        concept_id (str): The concept ID for the release
        use_sandbox (bool): Whether to use the Zenodo sandbox

    Returns:
        dict: The new version deposition
    """
    params = {"access_token": token_str}
    prefix = "sandbox." if use_sandbox else ""
    r = requests.post(
        f"https://{prefix}zenodo.org/api/deposit/depositions/{concept_id}/actions/newversion",
        params=params,
    )
    if r.status_code != 201:
        raise RuntimeError(f"Failed to create new version: {r.text}")

    new_version_link = r.json()["links"]["latest_draft"]

    r = requests.get(new_version_link, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to retrieve new version: {r.text}")

    return r.json()


def zenodo_delete_draft_files(token_str: str, draft: dict[str, Any]) -> None:
    """
    Remove all files from a Zenodo version draft.

    Args:
        token_str (str): The Zenodo access token
        draft (str): The Zenodo draft
        use_sandbox (bool): Whether to use the Zenodo sandbox
    """
    params = {"access_token": token_str}

    for f in draft["files"]:
        r = requests.delete(f["links"]["self"], params=params)
        if r.status_code != 204:
            raise RuntimeError(f"Failed to delete {f['filename']}: {r.text}")


def zenodo_upload_files(
    token_str: str, draft: dict[str, Any], files: list[str]
) -> None:
    """
    Upload files to a Zenodo version draft.

    Args:
        token_str (str): The Zenodo access token
        draft (str): The Zenodo draft
        files (list[str]): The files to upload
    """
    params = {"access_token": token_str}
    upload_bucket = draft["links"]["bucket"]

    for f in files:
        with open(f, "rb") as fp:
            r = requests.put(
                f"{upload_bucket}/{os.path.basename(f)}",
                data=fp,
                params=params,
            )
            if r.status_code != 201:
                raise RuntimeError(f"Failed to upload {f}: {r.text}")


def zenodo_update_metadata(
    token_str: str, draft: dict[str, Any], metadata: dict[str, Any]
) -> None:
    """
    Update the metadata for a Zenodo version draft.

    Args:
        token_str (str): The Zenodo access token
        draft_id (str): The draft ID
        metadata (dict[str, Any]): The metadata to update
    """
    params = {"access_token": token_str}
    draft_id = draft["id"]

    r = requests.put(
        f"https://sandbox.zenodo.org/api/deposit/depositions/{draft_id}",
        params=params,
        data=json.dumps(metadata),
    )

    if r.status_code != 200:
        raise RuntimeError(f"Failed to update metadata: {r.text}")


def zenodo_publish_release(token_str: str, draft: dict[str, Any]) -> None:
    """
    Publish a Zenodo release.

    Args:
        token_str (str): The Zenodo access token
        draft (str): The Zenodo draft
    """
    params = {"access_token": token_str}
    draft_id = draft["id"]

    r = requests.post(
        f"https://sandbox.zenodo.org/api/deposit/depositions/{draft_id}/actions/publish",
        params=params,
    )

    if r.status_code != 202:
        raise RuntimeError(f"Failed to publish release: {r.text}")


def create_zenodo_release(
    token: str = "",
    files: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    concept_id: str | None = None,
    publish: bool = False,
    use_sandbox: bool = True,
) -> bool:
    """
    Create a new documentation release on Zenodo.

    Args:
        token (str): The Zenodo access token
        files (list[str]): The files to upload
        metadata (dict[str, Any]): The metadata for the release
        concept_id (str): The concept ID for the release
        use_sandbox (bool): Whether to use the Zenodo sandbox
    """

    if files is None:
        raise ValueError("No files to upload")

    if metadata is None:
        raise ValueError("No metadata provided")

    if concept_id is None:
        raise ValueError("No concept ID provided")

    try:
        print("  Getting latest version...", end="\r")
        latest_id = zenodo_get_latest_version(token, concept_id, use_sandbox)
    except Exception as e:
        print(CROSS)
        print(f"Failed to get latest version: {e}")
        return False
    else:
        print(TICK)

    try:
        print("  Creating new version...", end="\r")
        draft = zenodo_create_new_version(token, latest_id, use_sandbox)
    except Exception as e:
        print(CROSS)
        print(f"Failed to create new version: {e}")
        return False
    else:
        print(TICK)

    try:
        print("  Deleting existing files...", end="\r")
        zenodo_delete_draft_files(token, draft)
    except Exception as e:
        print(CROSS)
        print(f"Failed to delete existing files: {e}")
        return False
    else:
        print(TICK)

    try:
        print("  Uploading files...", end="\r")
        zenodo_upload_files(token, draft, files)
    except Exception as e:
        print(CROSS)
        print(f"Failed to upload files: {e}")
        return False
    else:
        print(TICK)

    try:
        print("  Updating metadata...", end="\r")
        zenodo_update_metadata(token, draft, metadata)
    except Exception as e:
        print(CROSS)
        print(f"Failed to update metadata: {e}")
        return False
    else:
        print(TICK)

    if publish:
        try:
            print("  Publishing release...", end="\r")
            zenodo_publish_release(token, draft)
        except Exception as e:
            print(CROSS)
            print(f"Failed to publish release: {e}")
            return False
        else:
            print(TICK)

    return True


def main(
    no_interactive: bool = False,
    confirm_checks: bool = False,
    release_description: str = "",
    force_docs: bool = False,
    zenodo_publish: bool = False,
    zenodo_sandbox: bool = False,
) -> None:
    """
    The main function for the publish script. This function will create a new release
    of the ppodd software, tag the release in git, create a GitHub release, generate
    the documentation, and create a Zenodo release.
    """

    zenodo_token = get_zenodo_token()
    github_token = get_github_token()

    print()
    print("Creating new decades-ppandas (ppodd) release\n")

    repo_directory = os.path.abspath(
        os.path.join(os.path.dirname(ppodd.__file__), "..")
    )
    if os.getcwd() != repo_directory:
        print(f"Note: Moving into repository: {repo_directory}\n")
        os.chdir(repo_directory)

    if not no_interactive and not confirm_checks:
        print("Has this release been checked against existing data? [y/n]")
        checked = input("> ")
        confirm_checks = checked.lower() == "y"

    if not confirm_checks:
        print("Please check the data before proceeding.\n")
        sys.exit(1)

    if not no_interactive:
        print(f"This release will be tagged as v{ppodd.version()}. Continue? [y/n]")
        proceed = input("> ")
        if proceed.lower() != "y":
            print("Exiting without creating a release.\n")
            sys.exit(1)
    else:
        print(f"Creating release v{ppodd.version()}...")

    if get_patch_version() != "0":
        if not no_interactive:
            print("Patch release. Force generation of documentation? [y/n]")
            force_str = input("> ")
            gen_docs = force_str.lower() == "y"
        else:
            gen_docs = force_docs
    else:
        gen_docs = True

    if not no_interactive:
        print("Use Zenodo sandbox [y/n]")
        zenodo_str = input("> ")
        zenodo_sandbox = zenodo_str.lower() == "y"

        print("Publish Zenodo draft immediately [y/n]")
        zenodo_str = input("> ")
        zenodo_publish = zenodo_str.lower() == "y"

    try:
        tag_release()
    except TagError as e:
        print(f"Could not create tag {e.tag}. Git error: {e.giterr}\n")
        sys.exit(1)
    except GitHubError:
        print("Failed to push tag to remote. You may need to do this manually.\n")
        sys.exit(1)

    if not release_description and not no_interactive:
        print("Please enter a description for the release:")
        release_description = input("> ")

    try:
        print("  Creating GitHub release...", end="\r")
        create_git_release(release_description, github_token)
    except Exception as e:
        print(f"{CROSS}")
        print("Failed to create GitHub release: {e}")
        sys.exit(1)
    else:
        print(TICK)

    if gen_docs:
        print("  Generating documentation [web]...", end="\r")
        with ppodd.flipdir("ppodd/docs/data"):
            try:
                subprocess.run(["make", "publish"], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(CROSS)
                print(f"Failed to generate documentation: {e.stderr.decode()}")
            else:
                print(TICK)

        print("  Generating documentation [pdf]... ", end="\r")
        with ppodd.flipdir("ppodd/docs/data"):
            try:
                subprocess.run(["make", "latexpdf"], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(CROSS)
                print(f"Failed to generate PDF documentation: {e.stderr.decode()}")
            else:
                print(TICK)

        with ppodd.flipdir("ppodd/docs/data/_build/latex"):
            pdf_files = [f for f in os.listdir() if f.endswith(".pdf")]
            print("Creating Zenodo release...")
            success = create_zenodo_release(
                token=zenodo_token,
                files=pdf_files,
                metadata=ZENODO_METADATA,
                concept_id=ZENODO_CONCEPT_ID["sandbox" if zenodo_sandbox else "live"],
                publish=zenodo_publish,
                use_sandbox=zenodo_sandbox,
            )
            if not success:
                print(f"{CROSS} Failed to create Zenodo release.")


def get_parser() -> argparse.Namespace:
    """
    Argument parser for the publish script. This function will return an argument
    parser for the publish script.
    """
    parser = argparse.ArgumentParser(description="Publish a new release of ppodd")

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Do not prompt for user input",
    )

    parser.add_argument(
        "--confirm-checks",
        action="store_true",
        default=False,
        help="Indicate that the data have been checked (see ppodd.cli.check_version)",
    )

    parser.add_argument(
        "--release-description",
        type=str,
        help="The description for the release",
    )

    parser.add_argument(
        "--force-docs",
        action="store_true",
        default=False,
        help="Force generation of documentation for a patch release",
    )

    parser.add_argument(
        "--zenodo-publish",
        action="store_true",
        default=False,
        help="Publish the release to Zenodo rather than just creating a draft",
    )

    parser.add_argument(
        "--zenodo-sandbox",
        action="store_true",
        default=False,
        help="Use the Zenodo sandbox rather than the live site",
    )

    args = parser.parse_args()

    if args.no_interactive and not args.confirm_checks:
        raise ValueError("Cannot use --no-interactive without --confirm-checks")

    if args.no_interactive and not args.release_description:
        raise ValueError("Cannot use --no-interactive without --release-description")

    return args


if __name__ == "__main__":
    try:
        args = get_parser()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    main(**vars(args))
