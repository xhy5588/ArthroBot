"""Compare adjustable-rod browser results with independent NumPy FK/torques.

Run from the repository root:
    /home/eric/env_isaaclab/bin/python projects/arthro_bot/payload_tool/payload_viewer/check_rod_viewer.py
"""
import asyncio
import base64
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import qmc

from browser_check import BrowserCheck

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from assess_rod_payload import RodVariants


def expected_workspace(model, q):
    own, unit, _, tcp, pivots, _ = model.evaluate(q)
    denominator = np.where(np.abs(unit) > 1e-10, unit, 1.)
    upper = np.maximum((-model.caps-own)/denominator, (model.caps-own)/denominator)
    upper[np.abs(unit) <= 1e-10] = np.inf
    capacity = upper.min(axis=1)
    capacity[(np.abs(own) > model.caps).any(axis=1)] = -1
    down = (tcp[:, 2]-pivots[:, 5, 2])/np.linalg.norm(model.grasp) <= -np.cos(np.pi/12)
    ijk = np.floor((tcp+2.5)/.05).astype(int)
    keys = (ijk[:, 0]*100+ijk[:, 1])*100+ijk[:, 2]
    results = {}
    for name, valid in [('any_orientation', np.ones(len(q), bool)), ('downward_15deg', down)]:
        dense = np.full(100**3, -1.)
        np.maximum.at(dense, keys[valid], capacity[valid])
        results[name] = dict(cells=len(np.unique(keys[valid])),
                             loads=[int((dense >= load).sum()) for load in [0, .5, 1, 1.5, 2]])
    return results, (own, unit, tcp, capacity, down)


async def check():
    async with BrowserCheck(height=1450) as browser:
        command, evaluate, errors = browser.command, browser.evaluate, browser.errors
        await browser.open(HERE/'workspace_payload/payload_workspace.html',
                           'Boolean(window.workspace && workspace.setDesignLengths)')
        await evaluate("window.armEditor?.setMode('workspace')")
        variants = RodVariants()
        codes = np.floor(qmc.Sobol(6, scramble=True, seed=20261008).random_base2(18)*65536).astype(np.uint16)
        q = (codes.astype(float)+.5)/65536*2*np.pi-np.pi
        report = dict(passed=False, variants=[], browser_errors=errors)
        for lengths in ([100,100], [150,150], [650,650], [100,650], [750,50]):
            start = time.monotonic()
            result = await evaluate('''(async()=>{
                const result=await workspace.setDesignLengths(LENGTHS);
                const stats={};
                for(const [name,data] of Object.entries(result.datasets))stats[name]={
                  cells:data.n,loads:[0,.5,1,1.5,2].map(m=>Array.from(data.capacity).filter(x=>x>=m).length)};
                const data=result.datasets.any_orientation,selected=[];
                for(let i=0;i<100;i++){
                  const index=Math.floor(i*(data.n-1)/99);
                  selected.push({q:Array.from(data.angles.subarray(index*6,index*6+6)),
                    tcp:Array.from(data.xyz.subarray(index*3,index*3+3)),capacity:data.capacity[index]});
                }
                return {stats,probes:result.probes,selected,elapsedMs:result.elapsedMs,
                  model:workspace.getModel(),zero:workspace.kinematics([0,0,0,0,0,0]),
                  busy:document.getElementById('scene').dataset.busy};
            })()'''.replace('LENGTHS', json.dumps(lengths)))
            model = variants.model(np.array(lengths)/1000)
            for field in ['origins', 'masses', 'centers']:
                np.testing.assert_allclose(result['model'][field], getattr(model, field), atol=1e-12)
            expected, (own, unit, tcp, capacity, down) = expected_workspace(model, q)
            assert result['stats'] == expected, (lengths, result['stats'], expected)
            torque_error = 0.
            for probe in result['probes']:
                i = probe['index']
                np.testing.assert_allclose(probe['tcp'], tcp[i], atol=1e-12)
                np.testing.assert_allclose(probe['own'], own[i], atol=1e-10)
                np.testing.assert_allclose(probe['unit'], unit[i], atol=1e-10)
                assert probe['down'] == bool(down[i])
                np.testing.assert_allclose(probe['capacity'], capacity[i], atol=1e-8)
                torque_error = max(torque_error, np.abs(np.array(probe['own'])-own[i]).max())
            selected_q = np.array([p['q'] for p in result['selected']])
            so, su, _, stcp, *_ = model.evaluate(selected_q)
            np.testing.assert_allclose(stcp, [p['tcp'] for p in result['selected']], atol=1e-7)
            scap = model.capacity(selected_q)
            scap[(np.abs(so)>13).any(axis=1)] = -1
            np.testing.assert_allclose(scap, [p['capacity'] for p in result['selected']], atol=1e-8)
            zero = model.evaluate(np.zeros(6))
            np.testing.assert_allclose(result['zero']['own'], zero[0][0], atol=1e-10)
            np.testing.assert_allclose(result['zero']['unit'], zero[1][0], atol=1e-10)
            assert result['busy'] == 'false'
            item = dict(lengths_mm=lengths, samples=len(q), stats=expected,
                        max_probe_torque_error_nm=float(torque_error),
                        browser_calculation_s=result['elapsedMs']/1000,
                        check_elapsed_s=time.monotonic()-start)
            report['variants'].append(item)
            print(json.dumps(item), flush=True)
        interactions = await evaluate(r'''(async()=>{
          const assert=(x,message)=>{if(!x)throw Error(message);};
          await workspace.setDesignLengths([650,650]);workspace.updateLoad(2);
          const invalid=workspace.current.capacity.findIndex(x=>x<0);workspace.selectPose(invalid);
          assert(document.getElementById('pose-status').classList.contains('exceeded'),'Empty-arm overload not shown');
          document.getElementById('mode').value='downward_15deg';document.getElementById('mode').onchange();
          assert(workspace.current===workspace.state.designDatasets.downward_15deg,'Orientation changed back to original data');
          document.getElementById('mode').value='any_orientation';document.getElementById('mode').onchange();
          document.getElementById('equal-rods').checked=false;
          document.getElementById('upper-length').value=100;document.getElementById('upper-length').oninput();
          await new Promise(resolve=>setTimeout(resolve,230));
          while(document.getElementById('scene').dataset.busy==='true')await new Promise(resolve=>setTimeout(resolve,50));
          assert(workspace.design.lengths[0]===100&&workspace.design.lengths[1]===650,'Independent rod slider failed');
          const stale=workspace.setDesignLengths([75,75]);const latest=workspace.setDesignLengths([250,250]);
          await Promise.all([stale,latest]);assert(workspace.design.lengths[0]===250,'Stale worker result replaced latest model');
          workspace.showOriginal();assert(!workspace.state.design&&workspace.getVoxel()===.025,'Original map did not restore');
          assert(workspace.current===workspace.datasets.any_orientation,'Original map data did not restore');
          workspace.toggleLengthAnimation();
          while(document.getElementById('scene').dataset.busy==='true')await new Promise(resolve=>setTimeout(resolve,50));
          workspace.stopLengthAnimation();assert(!workspace.design.touring,'Length animation did not pause');
          assert(workspace.design.lengths[0]===100,'Length animation did not advance');
          await workspace.setDesignLengths([650,650]);workspace.updateLoad(2);workspace.setView('three');workspace.draw();
          assert(document.documentElement.scrollWidth<=innerWidth,'Desktop layout overflows');
          if(gl)assert(gl.getError()===gl.NO_ERROR,'WebGL error');
          return {overload:true,orientation:true,independent_sliders:true,latest_request:true,
                  original_map_restore:true,length_animation:true,desktop_layout:true};
        })()''')
        report['interaction_checks'] = interactions
        image = await command('Page.captureScreenshot', format='png', captureBeyondViewport=False)
        (HERE/'workspace_payload/rod_viewer_preview.png').write_bytes(base64.b64decode(image['data']))
        await command('Emulation.setDeviceMetricsOverride', width=412, height=1100, deviceScaleFactor=1, mobile=True)
        await asyncio.sleep(.3)
        mobile = await evaluate('({width:innerWidth,content:document.documentElement.scrollWidth,errors:workspace.diagnostics.errors})')
        assert mobile['content'] <= mobile['width'], mobile
        image = await command('Page.captureScreenshot', format='png', captureBeyondViewport=False)
        (HERE/'workspace_payload/rod_viewer_mobile.png').write_bytes(base64.b64decode(image['data']))
        report['mobile_layout'] = mobile
        assert not errors, errors
        assert not mobile['errors'], mobile
        report['passed'] = True
        (HERE/'workspace_payload/rod_viewer_validation.json').write_text(json.dumps(report, indent=2)+'\n')
        return report


if __name__ == '__main__':
    result = asyncio.run(check())
    print('PASS:', len(result['variants']), 'rod pairs; full workspace/torque comparison; browser controls; desktop/mobile layout.')
