"""Check browser filters and arm torques against the saved maps and Python model.

Runs in a temporary headless Chrome profile; it does not open or control the user's browser.


Run from the repository root:
    /home/eric/env_isaaclab/bin/python projects/arthro_bot/payload_tool/payload_viewer/check_viewer.py
"""
import asyncio
import json
from pathlib import Path
import sys

from browser_check import BrowserCheck

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from payload_analysis.paths import PROJECT_DIR, arm_urdf

HERE = PROJECT_DIR
from estimate_payload import ArmGravity


def main():
    output = HERE/'workspace_payload'
    cases, poses = [], []
    model = ArmGravity(arm_urdf())
    for mode in ('any_orientation', 'downward_15deg'):
        data = np.load(output/f'{mode}.npz')
        p, cap = data['sampled_tcp_m'], data['max_static_payload_kg']
        for region in ('all', 'above', 'slice'):
            in_height = np.ones(len(cap), bool) if region == 'all' else p[:, 2] >= 0 if region == 'above' else (p[:, 2] >= .1) & (p[:, 2] < .125)  # the one cell layer starting at the slider value, 10 cm
            for mass in (0., .5, 1., 1.5, 2.):
                valid = in_height & (cap >= mass)
                cases.append(dict(mode=mode, region=region, mass=mass, denominator=int(in_height.sum()),
                                  count=int(valid.sum()), radius=float(np.linalg.norm(p[valid, :2], axis=1).max())))
        ids = [0, 91, 704, len(cap)//2, len(cap)-1, int(cap.argmax())]
        for i in ids:
            q = data['joint_angles_rad'][i]
            own, unit, _, tcp, pivots, _ = model.evaluate(q)
            poses.append(dict(q=q.tolist(), own=own[0].tolist(), unit=unit[0].tolist(),
                              tcp=tcp[0].tolist(), pivots=pivots[0].tolist()))
    script = JS.replace('__CASES__', json.dumps(cases)).replace('__POSES__', json.dumps(poses))
    asyncio.run(check(script, output))


async def check(script, output):
    async with BrowserCheck() as browser:
        await browser.command('Page.addScriptToEvaluateOnNewDocument',
                              source='window.pageErrors=[];window.addEventListener("error",e=>pageErrors.push(e.message));')
        await browser.open(output/'payload_workspace.html', 'Boolean(window.workspace)')
        await browser.evaluate(script)
        await browser.wait_for('Boolean(document.getElementById("verification-result"))')
        result = json.loads(await browser.evaluate('document.getElementById("verification-result").textContent'))
        (output/'viewer_validation.json').write_text(json.dumps(result, indent=2)+'\n')
        if not result['passed']:
            raise AssertionError(result)
        if browser.errors:
            raise AssertionError(browser.errors)
        await browser.screenshot(output/'viewer_preview.png')
        print(json.dumps(result, indent=2))


JS = r'''
setTimeout(() => {
  const result = {passed:false};
  const assert = (condition, message) => {if(!condition)throw Error(message)};
  try {
    window.armEditor?.setMode('workspace');
    const w = window.workspace;
    assert(w, 'Viewer failed to initialize: '+JSON.stringify(window.pageErrors));
    const cases = __CASES__, poses = __POSES__;
    let maxPositionError=0,maxTorqueError=0;
    for(const example of poses){
      const actual=w.kinematics(example.q);
      maxPositionError=Math.max(maxPositionError,...actual.tcp.map((v,j)=>Math.abs(v-example.tcp[j])));
      for(const field of ['own','unit'])maxTorqueError=Math.max(maxTorqueError,...actual[field].map((v,j)=>Math.abs(v-example[field][j])));
      for(let i=0;i<6;i++)maxPositionError=Math.max(maxPositionError,...actual.pivots[i].map((v,j)=>Math.abs(v-example.pivots[i][j])));
    }
    assert(maxPositionError<1e-9,'Browser forward kinematics differs from the Python model');
    assert(maxTorqueError<1e-8,'Browser gravity torques differ from the Python model');
    result.max_position_error_m=maxPositionError;result.max_torque_error_nm=maxTorqueError;result.pose_checks=poses.length;
    for(const example of cases){
      document.getElementById('mode').value=example.mode;document.getElementById('mode').onchange();
      document.getElementById('height').value=example.region;document.getElementById('slice').value=.1;w.refreshScope();
      w.updateLoad(example.mass);
      assert(w.state.scope.length===example.denominator,'Height scope mismatch: '+JSON.stringify(example));
      assert(w.state.visible.length===example.count,'Payload filter mismatch: '+JSON.stringify(example));
      assert(Math.abs(parseFloat(document.getElementById('reach').textContent)-100*example.radius)<.051,'Reach statistic mismatch');
    }
    result.filter_checks=cases.length;
    document.getElementById('mode').value='downward_15deg';document.getElementById('mode').onchange();
    document.getElementById('height').value='slice';document.getElementById('slice').value=.7;w.refreshScope();
    assert(w.state.visible.length===0&&!document.getElementById('empty').hidden,'Empty slice is not represented correctly');
    assert(w.state.pose===null,'Out-of-slice selected pose was not cleared');result.empty_slice=true;
    document.getElementById('mode').value='any_orientation';document.getElementById('mode').onchange();
    document.getElementById('height').value='all';w.refreshScope();w.updateLoad(1);w.setView('top');w.draw();
    assert(Math.abs(w.state.pitch-Math.PI/2)<1e-12,'Top camera control failed');
    const index=w.state.visible.find(i=>w.current.neighbor[i]<1&&Math.abs(w.current.xyz[i*3])<.5);
    const p=w.project(Array.from(w.current.xyz.subarray(index*3,index*3+3)));
    assert(w.pickPoint(p[0],p[1])>=0,'Point inspection failed');
    const selected=w.state.selected,expected=Array.from(w.current.xyz.subarray(selected*3,selected*3+3));
    assert(Math.hypot(...w.state.pose.tcp.map((v,i)=>v-expected[i]))<1e-6,'Displayed arm misses the selected point');
    result.point_inspection=true;
    const weakest=w.current.capacity.indexOf(Math.min(...w.current.capacity));w.selectPose(weakest);w.updateLoad(2);
    assert(document.getElementById('pose-status').classList.contains('exceeded'),'Overloaded selected posture is not marked');
    result.overload_indication=true;
    w.updateLoad(1);w.toggleAnimation();w.state.phase=0;w.state.lastLoadUpdate=0;frame(performance.now()+200);
    assert(w.state.load<.1&&w.state.playing,'Payload animation did not advance');
    const poseBefore=w.state.selected;w.state.phase=Math.PI;w.state.lastLoadUpdate=0;frame(performance.now()+400);
    assert(w.state.load>1.9,'Payload animation did not reach the heavy-load region');
    assert(w.state.selected===poseBefore,'Payload animation unexpectedly changed the arm posture');
    w.stopAnimation();const stopped=w.state.load;frame(performance.now()+600);assert(w.state.load===stopped,'Pause did not stop the payload sweep');
    result.animation_and_pause=true;
    assert(document.documentElement.scrollWidth<=innerWidth,'Desktop layout overflows horizontally');
    if(gl)assert(gl.getError()===gl.NO_ERROR,'WebGL rendering error');
    assert(window.pageErrors.length===0,JSON.stringify(window.pageErrors));
    result.renderer=w.diagnostics.renderer;result.errors=window.pageErrors;result.source_samples=META.samples;
    // Leave the screenshot at an informative, reproducible view.
    w.updateLoad(1);w.state.selected=-1;w.refreshScope();w.setView('three');w.draw();
    result.passed=true;
  }catch(error){result.error=String(error);result.stack=error.stack;result.page_errors=window.pageErrors;}
  const output=document.createElement('pre');output.id='verification-result';output.hidden=true;output.textContent=JSON.stringify(result);document.body.appendChild(output);
},300);
'''


if __name__ == '__main__':
    main()
