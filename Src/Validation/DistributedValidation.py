"""Exercise the real HTTP worker protocol, native GPU tiles and PNG assembly."""
import argparse
import http.server
import json
from pathlib import Path
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib


def read_png(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'Invalid PNG signature'
    cursor, compressed, dimensions = 8, bytearray(), None
    while cursor < len(data):
        length = struct.unpack_from('>I', data, cursor)[0]
        kind = data[cursor + 4:cursor + 8]
        body = data[cursor + 8:cursor + 8 + length]
        crc = struct.unpack_from('>I', data, cursor + 8 + length)[0]
        assert zlib.crc32(kind + body) == crc, 'Invalid PNG chunk CRC'
        if kind == b'IHDR':
            width, height, depth, color, _, _, _ = struct.unpack('>IIBBBBB', body)
            assert depth == 8 and color == 6
            dimensions = width, height
        elif kind == b'IDAT':
            compressed.extend(body)
        cursor += length + 12
    assert dimensions
    width, height = dimensions
    filtered = zlib.decompress(compressed)
    stride = width * 4 + 1
    assert len(filtered) == stride * height
    assert all(filtered[y * stride] == 0 for y in range(height)), 'Unexpected PNG row filter'
    pixels = b''.join(filtered[y * stride + 1:(y + 1) * stride] for y in range(height))
    assert all(pixels[index] == 255 for index in range(3, len(pixels), 4))
    assert len(set(pixels[0::4])) > 8, 'Render is blank or degenerate'
    return dimensions, pixels


def request(url, path='/status', body=None, token='arvo-local-validation'):
    headers = {'Authorization': 'Bearer ' + token}
    req = urllib.request.Request(url + path, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', default='compute')
    parser.add_argument('--port', type=int, default=17831)
    parser.add_argument('--skip-4k', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    urls = [f'http://127.0.0.1:{args.port}', f'http://127.0.0.1:{args.port + 1}']
    for url in urls:
        deadline = time.monotonic() + 30
        while True:
            try:
                assert b'version=1' in request(url)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError('Worker did not become ready: ' + url)
                time.sleep(0.1)

    results = []

    def run(name, extra=(), expected=0, width=127, height=95, servers=urls):
        output = args.output / (name + '.png')
        report = args.output / (name + '.txt')
        if output.exists():
            output.unlink()
        command = [str(args.renderer), '--servers', ','.join(servers), '--backend', args.backend,
                   '--token', 'arvo-local-validation', '--scene', 'cornell', '--algorithm', 'path',
                   '--samples', '2', '--bounces', '2', '--width', str(width), '--height', str(height),
                   '--tile', '32', '--output', str(output), '--report', str(report), *extra]
        start = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (args.output / (name + '.log')).write_text(completed.stdout + completed.stderr)
        assert completed.returncode == expected, name + ': ' + completed.stdout + completed.stderr
        fields = dict(line.split('=', 1) for line in report.read_text().splitlines() if '=' in line)
        if expected == 0:
            dimensions, pixels = read_png(output)
            assert dimensions == (width, height), name + ': wrong output dimensions'
            assert fields['completed'] == fields['total'], name + ': incomplete tiles'
        else:
            assert not output.exists(), name + ': failed/cancelled image was published'
            pixels = None
        results.append({'test': name, 'seconds': round(time.monotonic() - start, 3), 'state': fields['state'],
                        'tiles': fields['completed'], 'servers': [fields.get(f'server{i}.completed') for i in range(len(servers))]})
        print('PASS', name, results[-1], flush=True)
        return fields, pixels

    for algorithm in ('ray', 'path'):
        _, full = run(algorithm + '-full', ['--algorithm', algorithm, '--tile', '128'], servers=urls[:1])
        fields, tiled = run(algorithm + '-tiled', ['--algorithm', algorithm])
        assert full == tiled, algorithm + ': full-frame and distributed tile pixels differ'
        assert all(int(fields[f'server{i}.completed']) > 0 for i in range(2)), 'Both worker processes must render tiles'

    run('sponza', ['--scene', 'sponza', '--algorithm', 'path', '--samples', '1'], width=127, height=73)
    run('failover', servers=[f'http://127.0.0.1:{args.port + 2}', *urls])
    run('all-unavailable', expected=1, servers=[f'http://127.0.0.1:{args.port + 2}'])
    run('unauthorized', ['--token', 'wrong'], expected=1)
    run('cancel', ['--samples', '64', '--bounces', '8', '--tile', '64', '--cancel-after-ms', '300'],
        expected=2, width=3840, height=2160)
    time.sleep(1)
    run('after-cancel', ['--algorithm', 'ray'])
    for url in urls:
        try:
            request(url, '/render', b'version=1\nscene=unknown\n')
            raise AssertionError('Malformed scene was accepted')
        except urllib.error.HTTPError as error:
            assert error.code == 400
    base_request = dict(version='1', job='protocol-validation', scene='cornell', backend=args.backend,
                        algorithm='ray', width='16', height='16', full_width='16', full_height='16',
                        x='0', y='0', samples='1', bounces='1', shadows='1', px='0', py='0', pz='4.2',
                        tx='0', ty='0', tz='0', ux='0', uy='1', uz='0', fov='60')
    for name, value in [('width', '0'), ('samples', '4097'), ('bounces', '33'), ('px', 'nan'),
                        ('fov', '180'), ('x', '16'), ('uy', '0'), ('backend', 'unavailable')]:
        invalid = dict(base_request, **{name: value})
        body = ''.join(f'{key}={value}\n' for key, value in invalid.items()).encode()
        try:
            request(urls[0], '/render', body)
            raise AssertionError('Invalid protocol field was accepted: ' + name)
        except urllib.error.HTTPError as error:
            assert error.code == 400

    class WrongSceneWorker(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'version=1\ncpu=-1\ngpu=-1\nbackend=test\n'
            self.send_response(200)
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            data = self.rfile.read(int(self.headers['Content-Length'])).decode()
            fields = dict(line.split('=', 1) for line in data.splitlines() if '=' in line)
            if self.path == '/cancel':
                self.server.cancelled.set()
                body = b'Cancelled'
            else:
                if self.server.slow:
                    self.server.cancelled.wait(5)
                width, height = int(fields['width']), int(fields['height'])
                body = b'ARV1' + struct.pack('<IIQQ', width, height, 0, 0) + bytes(width * height * 4)
            try:
                self.send_response(200)
                self.send_header('Content-Length', len(body))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *_):
            pass

    fake = http.server.ThreadingHTTPServer(('127.0.0.1', 0), WrongSceneWorker)
    fake.slow = False
    fake.cancelled = threading.Event()
    fake_thread = threading.Thread(target=fake.serve_forever, daemon=True)
    fake_thread.start()
    try:
        run('wrong-scene', expected=1, servers=[f'http://127.0.0.1:{fake.server_port}'])
        fake.slow = True
        fake.cancelled.clear()
        run('cancel-slow-worker', ['--cancel-after-ms', '300'], expected=2,
            servers=[f'http://127.0.0.1:{fake.server_port}'])
        assert fake.cancelled.wait(1), 'Cancellation was not delivered to the slow worker'
        assert results[-1]['seconds'] < 3, 'Slow peer blocked client cancellation'
    finally:
        fake.shutdown()
        fake.server_close()
        fake_thread.join()
    if not args.skip_4k:
        run('cornell-4k-path', ['--samples', '2', '--bounces', '2', '--tile', '256'], width=3840, height=2160)
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
    print('All distributed validation checks passed.', flush=True)


if __name__ == '__main__':
    main()
