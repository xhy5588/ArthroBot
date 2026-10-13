"""Small isolated Chrome/CDP harness shared by the offline viewer checks.

Only the temporary profile/process group created here is managed. The user's
open browser and Isaac Sim are never attached to or terminated.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import urllib.request

import websockets


class BrowserCheck:
    def __init__(self, width=1500, height=1260):
        self.width, self.height = width, height
        self.errors = []
        self.serial = 0
        self.ws = self.proc = self.log = self.directory = None

    async def __aenter__(self):
        chrome = shutil.which('google-chrome') or shutil.which('chromium')
        if not chrome:
            raise RuntimeError('Chrome or Chromium is required for viewer verification.')
        self.directory = tempfile.TemporaryDirectory(prefix='arm-payload-browser-')
        self.tmp = Path(self.directory.name)
        self.log = (self.tmp/'chrome.log').open('w')
        flags = ['--headless', '--no-sandbox', '--no-first-run', '--disable-gpu',
                 '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                 '--password-store=basic', '--use-mock-keychain',
                 '--disable-dev-shm-usage', '--no-proxy-server',
                 '--disable-background-networking', '--remote-debugging-port=0',
                 f'--user-data-dir={self.tmp/"profile"}',
                 f'--window-size={self.width},{self.height}']
        self.proc = subprocess.Popen([chrome, *flags, 'about:blank'],
                                     stdout=self.log, stderr=self.log, start_new_session=True)
        try:
            port_file = self.tmp/'profile/DevToolsActivePort'
            for _ in range(200):
                if port_file.exists():
                    break
                await asyncio.sleep(.1)
            if not port_file.exists():
                raise RuntimeError('Chrome did not start: '+(self.tmp/'chrome.log').read_text()[-2000:])
            port = port_file.read_text().splitlines()[0]
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=10) as response:
                pages = json.load(response)
            page = next(page for page in pages if page['type'] == 'page')
            self.ws = await websockets.connect(page['webSocketDebuggerUrl'], max_size=40*1024*1024)
            await self.command('Runtime.enable')
            await self.command('Page.enable')
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def command(self, method, **params):
        self.serial += 1
        await self.ws.send(json.dumps(dict(id=self.serial, method=method, params=params)))
        while True:
            response = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=45))
            if response.get('method') == 'Runtime.exceptionThrown':
                self.errors.append(response['params'])
            if response.get('id') == self.serial:
                if 'error' in response:
                    raise RuntimeError(response)
                return response.get('result', {})

    async def evaluate(self, expression):
        response = await self.command('Runtime.evaluate', expression=expression,
                                      awaitPromise=True, returnByValue=True)
        if 'exceptionDetails' in response:
            raise AssertionError(response['exceptionDetails'])
        return response['result'].get('value')

    async def wait_for(self, expression, timeout=20):
        for _ in range(int(timeout*10)):
            value = await self.evaluate(expression)
            if value:
                return value
            await asyncio.sleep(.1)
        raise TimeoutError(f'Browser condition did not become true: {expression}')

    async def open(self, path, ready):
        url = path.as_uri() if isinstance(path, Path) else path
        await self.command('Page.navigate', url=url)
        return await self.wait_for(ready)

    async def screenshot(self, path):
        result = await self.command('Page.captureScreenshot', format='png', captureBeyondViewport=False)
        path.write_bytes(base64.b64decode(result['data']))

    async def __aexit__(self, *unused):
        if self.ws:
            await self.ws.close()
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        if self.log:
            self.log.close()
        if self.directory:
            self.directory.cleanup()
