"""Verify a copied worker/client bundle locates its own assets from another cwd."""
import argparse
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

from DistributedValidation import read_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backend', default='compute')
    parser.add_argument('--port', type=int, default=17841)
    args = parser.parse_args()
    root = args.root.resolve()
    deploy = root / 'Intermediate' / 'Validation' / 'Deployment'
    deploy.mkdir(parents=True, exist_ok=True)
    suffix = '.exe' if (root / 'Binaries/Release/ArvoWorker.exe').exists() else ''
    for name in ('ArvoWorker', 'ArvoRender'):
        shutil.copy2(root / 'Binaries' / 'Release' / (name + suffix), deploy / (name + suffix))
    shutil.copytree(root / 'Contents', deploy / 'Contents', dirs_exist_ok=True)
    try:
        probe = socket.create_connection(('127.0.0.1', args.port), timeout=1)
    except OSError:
        pass
    else:
        probe.close()
        raise RuntimeError('Deployment test port is already occupied')
    command = [str(deploy / ('ArvoWorker' + suffix)), '--port', str(args.port), '--backend', args.backend, '--token', '']
    flags = subprocess.CREATE_NO_WINDOW if suffix else 0
    endpoint = f'http://127.0.0.1:{args.port}'
    with (deploy / 'worker.log').open('w') as output, (deploy / 'worker.err').open('w') as error:
        worker = subprocess.Popen(command, stdout=output, stderr=error, cwd=root.parent, creationflags=flags)
        try:
            deadline = time.monotonic() + 30
            while True:
                if worker.poll() is not None:
                    raise RuntimeError('Deployed worker exited: ' + (deploy / 'worker.err').read_text())
                try:
                    with urllib.request.urlopen(endpoint + '/status', timeout=1):
                        break
                except Exception:
                    if time.monotonic() > deadline:
                        raise RuntimeError('Deployed worker did not become ready')
                    time.sleep(0.1)
            log = (deploy / 'worker.log').read_text()
            assert str(deploy / 'Contents') in log, 'Worker selected the original compiled asset path'
            for scene in ('cornell', 'sponza'):
                image = deploy / (scene + '.png')
                completed = subprocess.run([str(deploy / ('ArvoRender' + suffix)), '--servers', endpoint,
                    '--backend', args.backend, '--scene', scene, '--width', '127', '--height', '73',
                    '--samples', '1', '--bounces', '2', '--output', str(image), '--token', ''], cwd=root.parent,
                    capture_output=True, text=True, timeout=120, creationflags=flags)
                (deploy / (scene + '.log')).write_text(completed.stdout + completed.stderr)
                assert completed.returncode == 0, completed.stdout + completed.stderr
                dimensions, _ = read_png(image)
                assert dimensions == (127, 73)
                print('PASS relocated bundle:', scene, flush=True)
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)


if __name__ == '__main__':
    main()
