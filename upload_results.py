import os

# import posixpath
from s3fs.core import S3FileSystem

s3 = S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")})

llm_used = os.environ.get("LLM")
remote_logs_folder_path = os.path.join(
    os.environ.get("AICHOR_OUTPUT_PATH"), llm_used + "_logs"
)
local_logs_folder = "./logs"

remote_workspace_folder_path = os.path.join(
    os.environ.get("AICHOR_OUTPUT_PATH"), llm_used + "_workspaces"
)
local_workspace_folder = "./workspaces"

print(f"Uploading logs to {remote_logs_folder_path}")

for root, dirs, files in os.walk(local_logs_folder):
    print(f"Uploading logs from {root}")

    for file in files:
        local_file_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_file_path, local_logs_folder)
        s3_file_path = os.path.join(remote_logs_folder_path, relative_path)
        # s3_file_path = posixpath.join(target_s3_path_base, relative_path)

        with open(local_file_path, "rb") as data:
            with s3.open(s3_file_path, "wb") as s3_file:
                s3_file.write(data.read())

print(f"Upload completed to {local_logs_folder}")


print(f"Uploading logs to {remote_workspace_folder_path}")

for root, dirs, files in os.walk(local_workspace_folder):
    print(f"Uploading workspace data from {root}")

    for file in files:
        local_file_path = os.path.join(root, file)
        relative_path = os.path.relpath(local_file_path, local_workspace_folder)
        s3_file_path = os.path.join(remote_workspace_folder_path, relative_path)
        # s3_file_path = posixpath.join(target_s3_path_base, relative_path)

        with open(local_file_path, "rb") as data:
            with s3.open(s3_file_path, "wb") as s3_file:
                s3_file.write(data.read())

print(f"Upload completed to {local_workspace_folder}")
