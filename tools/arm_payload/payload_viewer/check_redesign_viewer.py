"""Exercise the offline fixed-base comparison in an isolated Chrome profile.

Run from the repository root:
    /home/eric/env_isaaclab/bin/python projects/arthro_bot/payload_tool/payload_viewer/check_redesign_viewer.py
"""
import asyncio
import base64
import json
from pathlib import Path

from browser_check import BrowserCheck

HERE = Path(__file__).resolve().parent.parent


async def check():
    async with BrowserCheck(height=1500) as browser:
        command, evaluate, errors = browser.command, browser.evaluate, browser.errors
        await browser.open(HERE/'workspace_payload/redesign_options.html', 'Boolean(window.redesignLab)')
        result = await evaluate(r'''(async()=>{
          const w=window.redesignLab,$=id=>document.getElementById(id);
          const assert=(v,message)=>{if(!v)throw Error(message);};
          assert(w.state.conceptId==='serial_geared','Page should start with the fixed-base geared concept');
          assert(w.data.concepts.every(c=>c.id.startsWith('serial_')),'Non-fixed-base design remains in comparison');
          assert(w.data.assumptions.fixed_base_required===true,'Fixed-base constraint missing');
          assert(w.data.assumptions.shoulder_height_confirmed===false,'Mounting assumption must remain unconfirmed');
          const checked=[];
          for(const concept of w.data.concepts){
            assert(w.chooseConcept(concept.id),'Concept selection failed');
            assert($('geometry-result').textContent===concept.geometry.passed.toLocaleString()+' / '+concept.geometry.total.toLocaleString(),'Geometry totals differ from report');
            assert($('torque-result').textContent===concept.statics.passed.toLocaleString()+' / '+concept.statics.total.toLocaleString(),'Torque totals differ from report');
            const indices=[0,351,701,concept.cases.findIndex(c=>!c.reachable),
              concept.cases.findIndex(c=>c.reachable&&!c.torque_pass),concept.cases.findIndex(c=>c.torque_pass)].filter(i=>i>=0);
            for(const i of indices){
              w.chooseCase(i);const c=w.current();
              assert($('pose-status').classList.contains('bad')!==Boolean(c.reachable&&c.torque_pass),'Pose pass/fail color is wrong');
              assert($('case-slider').value===String(i),'Pose slider out of sync');
              assert($('orientation').value===c.orientation_label,'Orientation selector out of sync');
              if(!c.reachable)assert(!$('load-values').querySelector('.torque-fill'),'Unreachable target displays torque as feasible');
              if(c.reachable){
                const text=Array.from($('load-values').querySelectorAll('.torque-number')).map(e=>e.textContent);
                assert(text.length===6&&text.every((v,j)=>v===Math.abs(c.torques_nm[j]).toFixed(1)),'Rendered torque differs from report');
                const bars=Array.from($('load-values').querySelectorAll('.torque-fill'));
                assert(bars.length===6&&bars.every((b,j)=>Math.abs(parseFloat(b.style.width)-Math.min(100,Math.abs(c.torques_nm[j])/concept.statics.caps_nm[j]*100))<.001),'Torque bar uses wrong joint cap');
                assert(bars.every((b,j)=>b.classList.contains('over')===(Math.abs(c.torques_nm[j])>concept.statics.caps_nm[j])),'Torque overload indicator uses wrong joint cap');
              }
            }
            checked.push({id:concept.id,casesChecked:indices.length,casesAvailable:concept.cases.length});
          }
          w.chooseConcept('serial_geared');
          for(let j=0;j<3;j++){const input=$('target-'+j);input.value=j===0?2:0;input.dispatchEvent(new Event('input'));}
          assert(w.current().position.every((x,j)=>Math.abs(x-[1.1,-.5,-.5][j])<1e-10),'Target sliders did not select requested position');
          $('orientation').value='Approach toward base · roll 90°';$('orientation').dispatchEvent(new Event('change'));
          assert(w.current().orientation_label.includes('toward base'),'Required orientation selection failed');
          w.chooseConcept('serial_long');assert(w.current().position[0]===1.1,'Changing concept did not preserve the target');
          const before=w.state.caseIndex;w.play();await new Promise(resolve=>setTimeout(resolve,950));w.stop();
          assert(w.state.caseIndex!==before&&!w.state.playing,'Discrete pose playback or pause failed');
          assert(document.documentElement.scrollWidth<=innerWidth,'Desktop overflow');
          const main=document.querySelector('main').textContent;
          assert(main.includes('not')&&main.includes('mass'),'Validation limits or mass assumptions missing');
          w.chooseConcept('serial_geared');w.selectTarget([.6,0,0],'Approach −Z · roll 0°');w.setView('perspective');w.draw();
          return {passed:true,concepts:checked,target_controls:true,orientation_controls:true,pose_playback:true,
                  report_values_match:true,desktop_layout:true,displayed_joint_torques:$('load-values').textContent};
        })()''')
        image = await command('Page.captureScreenshot', format='png', captureBeyondViewport=False)
        (HERE/'workspace_payload/redesign_preview.png').write_bytes(base64.b64decode(image['data']))
        await command('Emulation.setDeviceMetricsOverride', width=412, height=1100, deviceScaleFactor=1, mobile=True)
        await asyncio.sleep(.3)
        mobile = await evaluate('({width:innerWidth,content:document.documentElement.scrollWidth})')
        assert mobile['content'] <= mobile['width'], mobile
        result['mobile_layout'] = mobile
        image = await command('Page.captureScreenshot', format='png', captureBeyondViewport=False)
        (HERE/'workspace_payload/redesign_mobile.png').write_bytes(base64.b64decode(image['data']))
        assert not errors, errors
        result['browser_errors'] = errors
        (HERE/'workspace_payload/redesign_browser_validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(check())
