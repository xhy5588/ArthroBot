"""Verify the on-page worked examples against Python, plus layout and isolation.

Run from the repository root:
    /home/eric/env_isaaclab/bin/python projects/arthro_bot/payload_tool/payload_viewer/check_methods.py
"""
import asyncio
import json
from pathlib import Path
import sys

import numpy as np

from browser_check import BrowserCheck

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from payload_analysis.paths import PROJECT_DIR, arm_urdf

HERE = PROJECT_DIR
from estimate_payload import ArmGravity
from payload_analysis.page_data import worked_examples


async def check():
    model = ArmGravity(arm_urdf())
    analysis = worked_examples(HERE, model)
    probes = []
    for name, example in analysis['examples'].items():
        own, unit, *_ = model.evaluate(np.deg2rad(example['joint_angles_deg']))
        for mass in (0., .48, .5, 1., 2., example['payload_kg']):
            total = own[0]+mass*unit[0]
            probes.append(dict(pose=name, mass=mass, total=total.tolist(),
                               remaining=(model.caps-np.abs(total)).tolist(),
                               capacity=example['payload_kg'],
                               passed=bool((np.abs(total) <= model.caps+1e-10).all())))
    async with BrowserCheck(height=1200) as browser:
        await browser.open(HERE/'workspace_payload/payload_workspace.html',
                           'Boolean(window.ArmPayloadMethods && ArmPayloadMethods.lastResult)')
        await browser.evaluate('''(() => {
          document.documentElement.style.scrollBehavior='auto';
          const panel=document.getElementById('calculation-panel');
          if(!panel||panel.open)throw Error('Calculation disclosure must start collapsed');
          panel.open=true;
        })()''')
        await asyncio.sleep(.1)
        expression = r'''(() => {
          const check=(value,message)=>{if(!value)throw Error(message);};
          const api=ArmPayloadMethods, $=id=>document.getElementById(id);
          const before={load:workspace.state.load,design:workspace.state.design,selected:workspace.state.selected};
          check(!workspace.state.design, 'The page must start with the original arm');
          check(META.analysis.rod_lengths_mm.every(x=>x===150),'Worked examples use wrong rods');
          let error=0;
          for (const probe of PROBES) {
            const result=api.calculate(META.analysis.examples[probe.pose],probe.mass,META.analysis.motor_caps_nm);
            for(let j=0;j<6;j++) {
              error=Math.max(error,Math.abs(result.joints[j].total-probe.total[j]),
                                  Math.abs(result.joints[j].remaining-probe.remaining[j]));
            }
            check(result.pass===probe.passed,'Pass/fail differs from Python');
            check(Math.abs(result.interval.upper-probe.capacity)<1e-9,'Payload ceiling differs from Python');
          }
          check(error<1e-9,'Browser torque differs from Python');
          const cancellation=api.feasibleInterval([20],[-10],[13]);
          check(cancellation.feasible&&Math.abs(cancellation.lower-.7)<1e-12&&Math.abs(cancellation.upper-3.3)<1e-12,'Signed counterbalance interval lost');
          check(!api.feasibleInterval([14],[0],[13]).feasible,'Impossible zero-slope load treated as feasible');
          $('method-pose').value='extended';$('method-pose').dispatchEvent(new Event('change'));
          $('method-payload').value=.48;$('method-payload').dispatchEvent(new Event('input'));
          check($('method-load-status').dataset.pass==='true','0.48 kg should pass the example');
          document.querySelector('[data-method-load="0.5"]').click();
          check($('method-load-status').dataset.pass==='false','0.5 kg should exceed the extended pose');
          check($('method-joint-rows').children.length===6,'Six joint rows are required');
          check($('method-joint-rows').children[1].dataset.over==='true','J2 overload not identified');
          check($('method-capacity').textContent==='0.487 kg','Rounded extended ceiling is wrong');
          check($('method-substitution-equation').textContent.includes('9.9368'),'Worked substitution missing actual own-arm torque');
          check(workspace.state.load===before.load&&workspace.state.design===before.design&&workspace.state.selected===before.selected,'Example controls changed the explorer');
          check(new Set([...document.querySelectorAll('[id]')].map(x=>x.id)).size===document.querySelectorAll('[id]').length,'Duplicate element IDs');
          for(const a of document.querySelectorAll('a[href^="#"]')) {
            check(document.getElementById(a.hash.slice(1)),'Broken section anchor: '+a.hash);
          }
          check(document.documentElement.scrollWidth<=innerWidth,'Desktop overflow');
          check(document.querySelectorAll('script[src],link[rel="stylesheet"]').length===0,'Offline page added an external runtime dependency');
          $('calculation').scrollIntoView({behavior:'instant',block:'start'});
          return {passed:true,examples:PROBES.length,max_torque_error_nm:error,signed_intervals:true,
                  controls:true,explorer_unchanged:true,section_links:true,desktop_layout:true,calculation_disclosure:true};
        })()'''.replace('PROBES', json.dumps(probes))
        result = await browser.evaluate(expression)
        await asyncio.sleep(.2)
        await browser.screenshot(HERE/'workspace_payload/calculation_preview.png')
        await browser.command('Emulation.setDeviceMetricsOverride', width=412, height=1100,
                              deviceScaleFactor=1, mobile=True)
        await browser.evaluate('document.getElementById("calculation-panel").open=true;document.getElementById("method-example-heading").scrollIntoView({behavior:"instant"})')
        await asyncio.sleep(.2)
        result['mobile_layout'] = await browser.evaluate('({width:innerWidth,content:document.documentElement.scrollWidth,controls_visible:document.getElementById("method-pose").getBoundingClientRect().width>0&&document.getElementById("method-payload").getBoundingClientRect().width>0})')
        assert result['mobile_layout']['content'] <= result['mobile_layout']['width'] and result['mobile_layout']['controls_visible'], result
        await browser.screenshot(HERE/'workspace_payload/calculation_mobile.png')
        assert not browser.errors, browser.errors
        result['browser_errors'] = browser.errors
        (HERE/'workspace_payload/methods_validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(check())
