import paramiko
from scp import SCPClient
import os
import argparse
import time
import stat

# FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# TEST_DIR = os.path.join(FILE_DIR, "..", "test")
# MODEL_DIR = os.path.join(FILE_DIR, "..", "model")
BASE_DIR = "/home/jialuyu/Data_Final_Project/Remote_code"

# Define subdirectories
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")
HOME_DIR = os.environ["HOME"]
REMOTE_HOME_DIR = "/home/stella"
REMOTE_DIR = os.path.join(REMOTE_HOME_DIR, "Revised_James_Code/final_project_mVAE_pipeline/")
REMOTE_TEST_DIR = os.path.join(REMOTE_DIR, "test")
REMOTE_MODEL_DIR = os.path.join(REMOTE_DIR, "models_partdata_40000")

def download_file_with_key(
    host,
    port,
    username,
    private_key_path,
    passphrase,  # Add passphrase parameter
    remote_file_path,
    local_file_path,
):
    ssh = None
    try:
        # Load the private key with the passphrase
        private_key = paramiko.Ed25519Key.from_private_key_file(private_key_path, password=passphrase)

        # Establish an SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=username, pkey=private_key)

        # Use SFTP to check if it's a directory or a file
        sftp = ssh.open_sftp()
        try:
            remote_file_attr = sftp.stat(remote_file_path)
            # Check if it's a directory
            if stat.S_ISDIR(remote_file_attr.st_mode):
                # It's a directory, so use SCP with recursive flag
                with SCPClient(ssh.get_transport()) as scp:
                    scp.get(remote_file_path, local_file_path, recursive=True)
                print(f"Directory downloaded successfully to {local_file_path}")
            else:
                # It's a file, so use SCP without recursion
                with SCPClient(ssh.get_transport()) as scp:
                    scp.get(remote_file_path, local_file_path)
                print(f"File downloaded successfully to {local_file_path}")
        except FileNotFoundError:
            print(f"Error: The remote path {remote_file_path} was not found.")
        finally:
            sftp.close()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ssh:  # Only close ssh if it was successfully created
            print(f"Closing ssh connection to {host}\n")
            ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="lamb",
        choices=["lamb", "sheep"],
        help="The device to download model from",
    )
    parser.add_argument(
        "--username",
        default="stella",
        help="Username of the ssh server",
    )
    parser.add_argument(
        "--private-key",
        default="id_ed25519",
        help="File name of the private key",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Output model filename",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--passphrase",
        required=True,
        help="Passphrase for the private key",
    )

    args = parser.parse_args()

    device = args.device
    host = f"{device}.mech.northwestern.edu"
    port = 22
    username = args.username
    private_key_path = os.path.join(HOME_DIR, ".ssh", args.private_key)
    passphrase = args.passphrase  # Get the passphrase
    csv_file = args.csv
    model_file = args.model

    while True:
        download_file_with_key(
            host=host,
            port=port,
            username=username,
            private_key_path=private_key_path,  # Path to your private key
            passphrase=passphrase,  # Passphrase
            remote_file_path=os.path.join(REMOTE_TEST_DIR, csv_file),
            local_file_path=os.path.join(TEST_DIR, csv_file),
        )

        download_file_with_key(
            host=host,
            port=port,
            username=username,
            private_key_path=private_key_path,  # Path to your private key
            passphrase=passphrase,  # Passphrase
            remote_file_path=os.path.join(REMOTE_MODEL_DIR, model_file),
            local_file_path=os.path.join(MODEL_DIR, model_file),
        )

        time.sleep(2)
