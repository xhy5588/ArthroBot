"""Exercise the direct arm editor in an isolated Chrome/CDP session.

Includes real mouse drags on translation arrows and local rotation rings,
keyboard/input controls, invalid-target handling, map-pose adoption, and layout
checks. Numerical solver correctness is checked separately by check_pose_solver.


Run from the repository root:
    /home/eric/env_isaaclab/bin/python projects/arthro_bot/payload_tool/payload_viewer/check_manipulator.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import traceback

import numpy as np

from browser_check import BrowserCheck

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent/'workspace_payload'
# Verified nonsingular local posture, with a visible gripper in the arm view.
SEED = [0.4254273437044993, 0.9012462283176552, 0.10309068435135815,
        -0.8320477795370722, 0.8724554273816456, 0.7099364063280613]
STATE = r'''(() => {
  const e=window.armEditor,s=e.state;
  return {q:s.q, p:s.pose.tcp, R:s.pose.R, targetP:s.targetP,targetR:s.targetR,
    matched:s.matched,pending:s.pending,positionError:s.positionError,
    orientationError:s.orientationError,lastSolveMs:s.lastSolveMs,
    cap:document.getElementById('arm-capacity').textContent,
    status:document.getElementById('arm-status').textContent,
    note:document.getElementById('arm-target-note').textContent,
    mode:s.mode,tool:s.tool,drag:s.drag?{kind:s.drag.kind,index:s.drag.index}:null};
})()'''


def require(condition, message, context=None):
    if not condition:
        raise AssertionError(message + (': '+json.dumps(context) if context is not None else ''))


async def frame(browser):
    await browser.evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')


async def visible_rect(browser, selector):
    return await browser.evaluate('''(() => {
      const element=document.querySelector('''+json.dumps(selector)+''');
      if(!element)throw Error('Missing control: '+'''+json.dumps(selector)+''');
      for(let ancestor=element.parentElement;ancestor;ancestor=ancestor.parentElement){
        if(ancestor.tagName==='DETAILS')ancestor.open=true;
      }
      element.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'});
      const r=[...element.getClientRects()].find(rect=>rect.width>0&&rect.height>0)||element.getBoundingClientRect();
      return {x:r.x,y:r.y,width:r.width,height:r.height};
    })()''')


async def mouse(browser, event_type, x, y, pressed=False):
    params = dict(type=event_type, x=float(x), y=float(y),
                  button='left' if event_type != 'mouseMoved' else 'none',
                  buttons=1 if pressed else 0)
    if event_type in ('mousePressed', 'mouseReleased'):
        params['clickCount'] = 1
    await browser.command('Input.dispatchMouseEvent', **params)


async def click(browser, selector):
    rect = await visible_rect(browser, selector)
    x, y = rect['x']+rect['width']/2, rect['y']+rect['height']/2
    await mouse(browser, 'mouseMoved', x, y)
    await mouse(browser, 'mousePressed', x, y, True)
    await mouse(browser, 'mouseReleased', x, y)
    await frame(browser)


async def fill_number(browser, selector, value):
    await click(browser, selector)
    await browser.evaluate('document.querySelector('+json.dumps(selector)+').select()')
    await browser.command('Input.insertText', text=str(value))
    await browser.command('Input.dispatchKeyEvent', type='keyDown', key='Tab', code='Tab', windowsVirtualKeyCode=9)
    await browser.command('Input.dispatchKeyEvent', type='keyUp', key='Tab', code='Tab', windowsVirtualKeyCode=9)
    await frame(browser)


async def prepare_pose(browser, tool='move'):
    await browser.evaluate('''(() => {
      armEditor.setMode('arm');armEditor.setQ('''+json.dumps(SEED)+''');
      armEditor.setView('three');armEditor.setTool('''+json.dumps(tool)+''');
      armEditor.setLoad(1);armEditor.draw();
    })()''')
    await visible_rect(browser, '#arm-stage')
    await frame(browser)
    await browser.evaluate('armEditor.draw()')


async def perform_drag(browser, gesture, expected_kind, expected_index):
    rect = await visible_rect(browser, '#arm-stage')
    start = np.array(gesture['start']) + [rect['x'], rect['y']]
    end = np.array(gesture['end']) + [rect['x'], rect['y']]
    await mouse(browser, 'mouseMoved', *start)
    await mouse(browser, 'mousePressed', *start, True)
    pressed = await browser.evaluate(STATE)
    require(pressed['drag'] and pressed['drag']['kind'] == expected_kind,
            'Pointer did not grab the requested handle', pressed)
    require(pressed['drag']['index'] == expected_index, 'Pointer grabbed the wrong axis', pressed)
    for fraction in np.linspace(.2, 1., 5):
        position = start + fraction*(end-start)
        await mouse(browser, 'mouseMoved', *position, True)
        await frame(browser)
    await mouse(browser, 'mouseReleased', *end)
    await frame(browser)
    result = await browser.evaluate(STATE)
    require(not result['pending'] and result['drag'] is None, 'Drag did not finish', result)
    require(result['matched'], 'Known nearby drag target was not reached', result)
    require(result['positionError'] <= .001 and result['orientationError'] <= np.pi/180,
            'Editor marked a pose matched outside its tolerances', result)
    require(result['cap'] != '—', 'Reached pose has no payload readout', result)
    return result


TRANSLATE_GESTURE = r'''(() => {
  const e=armEditor,s=e.state,offset=.008;
  for(const h of s.handles.filter(h=>h.kind==='translate')){
    for(const fraction of [.72,.87,.95]){
      const start=h.start.slice(0,2).map((v,i)=>v+fraction*(h.end[i]-v));
      const hit=e.hitTest(start);
      const target=s.pose.tcp.map((v,i)=>v+h.axis[i]*offset);
      const p=e.project(s.pose.tcp),t=e.project(target);
      const end=start.map((v,i)=>v+t[i]-p[i]);
      if(hit&&hit.kind==='translate'&&hit.index===h.index&&
        [start,end].every(p=>p[0]>15&&p[0]<s.width-15&&p[1]>45&&p[1]<s.height-35)){
        return {start,end,index:h.index,axis:h.axis,offset,initialP:s.pose.tcp,initialR:s.pose.R};
      }
    }
  }
  throw Error('No isolated visible translation arrow available for test');
})()'''

ROTATE_GESTURE = r'''(() => {
  const e=armEditor,s=e.state,angle=.09;
  const add=(a,b)=>a.map((v,i)=>v+b[i]),mul=(a,k)=>a.map(v=>v*k);
  for(const h of s.handles.filter(h=>h.kind==='rotate').reverse()){
    const point=a=>e.project(add(s.pose.tcp,add(mul(h.u,h.radius*Math.cos(a)),mul(h.v,h.radius*Math.sin(a))))).slice(0,2);
    for(let i=4;i<77;i+=3){
      const a=i*Math.PI*2/80,start=point(a),end=point(a+angle),hit=e.hitTest(start);
      if(hit&&hit.kind==='rotate'&&hit.index===h.index&&Math.hypot(end[0]-start[0],end[1]-start[1])>3&&
        [start,end].every(p=>p[0]>15&&p[0]<s.width-15&&p[1]>45&&p[1]<s.height-35)){
        const targetR=ArmPoseSolver.multiply3(ArmPoseSolver.axisRotation(h.axis,angle),s.pose.R);
        return {start,end,index:h.index,axis:h.axis,angle,initialP:s.pose.tcp,initialR:s.pose.R,targetR};
      }
    }
  }
  throw Error('No isolated visible rotation-ring segment available for test');
})()'''


async def check_dialog_fit(browser):
    result = await browser.evaluate('''(() => {
      const d=document.getElementById('geometry-dialog'),box=d.getBoundingClientRect();
      const controls=[...d.querySelectorAll('input,button,select,a')]
        .map(element=>({id:element.id||element.textContent.trim(),rect:element.getBoundingClientRect()}))
        .filter(x=>x.rect.width>0&&x.rect.height>0);
      return {open:d.open,width:box.width,left:box.left,right:box.right,top:box.top,bottom:box.bottom,viewport:innerWidth,viewport_height:innerHeight,
        horizontal_overflow:d.scrollWidth>d.clientWidth+1,
        visible_controls:controls.length,
        controls_inside:controls.every(x=>x.rect.left>=box.left&&x.rect.right<=box.right),
        rod_fields_visible:['upper-length','forearm-length','upper-number','forearm-number']
          .every(id=>document.getElementById(id).getBoundingClientRect().width>0),
        focus_inside:d.contains(document.activeElement)};
    })()''')
    require(result['open'] and result['width']>250 and result['left']>=0 and result['right']<=result['viewport'] and result['top']>=0 and result['bottom']<=result['viewport_height'],
            'Geometry dialog does not fit the viewport', result)
    require(not result['horizontal_overflow'] and result['controls_inside'] and result['rod_fields_visible'],
            'Visible rod controls overflow or remain hidden', result)
    require(result['focus_inside'], 'Modal focus escaped the dialog', result)
    return result


async def run(output):
    report = dict(passed=False, checks={}, screenshots=[], browser_errors=[])
    output.mkdir(parents=True, exist_ok=True)
    async with BrowserCheck(width=1500, height=900) as browser:
        try:
            await browser.open(output/'payload_workspace.html', 'Boolean(window.armEditor && armEditor.state.pose)')
            await browser.command('Emulation.setDeviceMetricsOverride', width=1500, height=900, deviceScaleFactor=1, mobile=False)
            await browser.evaluate("document.documentElement.style.scrollBehavior='auto';document.body.style.scrollBehavior='auto'")
            await frame(browser)
            initial = await browser.evaluate(STATE)
            require(initial['mode'] == 'arm' and initial['matched'], 'Editor did not initialize at the actual arm pose', initial)
            require(initial['cap'] == '1.745 kg', 'Initial arm capacity differs from verified zero pose', initial)
            report['checks']['initial_zero_pose'] = dict(capacity=initial['cap'])

            initial_layout = await browser.evaluate('''(() => {
              const canvas=document.getElementById('arm-canvas').getBoundingClientRect();
              const capacity=document.getElementById('arm-capacity').getBoundingClientRect();
              const advanced=[...document.querySelectorAll('#arm-editor details')];
              return {viewport:[innerWidth,innerHeight],canvas_top:canvas.top,canvas_height:canvas.height,
                capacity_top:capacity.top,capacity_bottom:capacity.bottom,capacity_width:capacity.width,
                advanced_count:advanced.length,advanced_closed:advanced.every(d=>!d.open),
                calculation_closed:!document.getElementById('calculation-panel').open,
                assumptions_closed:!document.getElementById('assumptions-panel').open};
            })()''')
            # The Draft-style title and section links sit above the calculator; the arm
            # still starts in the top third of a 900 px-high window, fully in view.
            require(0<=initial_layout['canvas_top']<340 and initial_layout['canvas_height']>300
                    and initial_layout['canvas_top']+initial_layout['canvas_height']<=initial_layout['viewport'][1]+40,
                    'The initial arm canvas is not immediately visible near the top', initial_layout)
            require(initial_layout['capacity_width']>0 and 0<=initial_layout['capacity_top']
                    and initial_layout['capacity_bottom']<=initial_layout['viewport'][1], 'Live capacity is below the initial fold', initial_layout)
            require(initial_layout['advanced_count']>0 and initial_layout['advanced_closed'],
                    'Advanced arm controls must start collapsed', initial_layout)
            require(initial_layout['calculation_closed'] and initial_layout['assumptions_closed'],
                    'Reference sections must start collapsed', initial_layout)
            report['checks']['initial_simple_layout'] = initial_layout

            # Native modal controls must open explicitly, fit, and close by all
            # advertised routes, including the platform Escape behavior.
            await click(browser, '#open-geometry')
            require(await browser.evaluate("document.getElementById('geometry-dialog').open"), 'Rod dialog did not open')
            await click(browser, '#design-map')
            await browser.wait_for('workspace.state.design && !workspace.design.worker')
            report['checks']['desktop_geometry_dialog'] = await check_dialog_fit(browser)
            await click(browser, '#close-geometry')
            require(not await browser.evaluate("document.getElementById('geometry-dialog').open"), 'Rod dialog close button failed')
            await click(browser, '#open-geometry')
            await click(browser, '#done-geometry')
            require(not await browser.evaluate("document.getElementById('geometry-dialog').open"), 'Rod dialog Done button failed')
            await click(browser, '#open-geometry')
            await browser.command('Input.dispatchKeyEvent', type='keyDown', key='Escape', code='Escape', windowsVirtualKeyCode=27)
            await browser.command('Input.dispatchKeyEvent', type='keyUp', key='Escape', code='Escape', windowsVirtualKeyCode=27)
            await frame(browser)
            require(not await browser.evaluate("document.getElementById('geometry-dialog').open"), 'Native Escape did not close the rod dialog')
            report['checks']['geometry_dialog_close_routes'] = ['close button','Done','Escape']
            await browser.evaluate('workspace.showOriginal()')

            await click(browser, 'a[href="#calculation"]')
            require(await browser.evaluate("document.getElementById('calculation-panel').open"), 'Calculation anchor did not reveal its disclosure')
            await click(browser, 'a[href="#assumptions"]')
            require(await browser.evaluate("document.getElementById('assumptions-panel').open"), 'Assumptions anchor did not reveal its disclosure')
            report['checks']['reference_hash_navigation'] = dict(calculation=True,assumptions=True)
            await browser.evaluate("document.getElementById('calculation-panel').open=false;document.getElementById('assumptions-panel').open=false;history.replaceState(null,'',location.href.split('#')[0]);scrollTo(0,0)")

            await prepare_pose(browser, 'move')
            translation = await browser.evaluate(TRANSLATE_GESTURE)
            translated = await perform_drag(browser, translation, 'translate', translation['index'])
            expected_p = np.asarray(translation['initialP']) + .008*np.asarray(translation['axis'])
            np.testing.assert_allclose(translated['targetP'], expected_p, atol=2e-5, rtol=0)
            np.testing.assert_allclose(translated['targetR'], translation['initialR'], atol=1e-12, rtol=0)
            require(np.linalg.norm(np.asarray(translated['p'])-translation['initialP']) > .004,
                    'Mouse translation did not move the actual arm', translated)
            report['checks']['mouse_translation_arrow'] = dict(axis=translation['index'], requested_m=.008,
                position_error_m=translated['positionError'], orientation_error_rad=translated['orientationError'],
                final_solve_ms=translated['lastSolveMs'])

            await prepare_pose(browser, 'rotate')
            rotation = await browser.evaluate(ROTATE_GESTURE)
            rotated = await perform_drag(browser, rotation, 'rotate', rotation['index'])
            np.testing.assert_allclose(rotated['targetP'], rotation['initialP'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(rotated['targetR'], rotation['targetR'], atol=2e-3, rtol=0)
            require(np.max(np.abs(np.asarray(rotated['R'])-rotation['initialR'])) > .02,
                    'Mouse rotation did not rotate the actual gripper', rotated)
            report['checks']['mouse_local_rotation_ring'] = dict(axis=rotation['index'], requested_rad=.09,
                position_error_m=rotated['positionError'], orientation_error_rad=rotated['orientationError'],
                final_solve_ms=rotated['lastSolveMs'])

            # Native keyboard stepping exercises the actual range input event.
            await prepare_pose(browser)
            before = await browser.evaluate("Number(document.getElementById('arm-joint-1').value)")
            await visible_rect(browser, '#arm-joint-1')
            await browser.evaluate("document.getElementById('arm-joint-1').focus()")
            await browser.command('Input.dispatchKeyEvent', type='keyDown', key='ArrowRight', code='ArrowRight', windowsVirtualKeyCode=39)
            await browser.command('Input.dispatchKeyEvent', type='keyUp', key='ArrowRight', code='ArrowRight', windowsVirtualKeyCode=39)
            await frame(browser)
            after = await browser.evaluate("Number(document.getElementById('arm-joint-1').value)")
            joint = await browser.evaluate(STATE)
            require(after > before and joint['matched'], 'Joint slider did not update the pose', joint)
            np.testing.assert_allclose(joint['q'][1], np.deg2rad(after), atol=1e-12)
            report['checks']['joint_slider_keyboard'] = dict(before_deg=before, after_deg=after)

            await prepare_pose(browser)
            numeric_start = await browser.evaluate(STATE)
            new_x = round(numeric_start['p'][0]+.004, 6)
            await fill_number(browser, '#arm-target-x', new_x)
            numeric = await browser.evaluate(STATE)
            require(numeric['matched'], 'Numeric position edit failed for a nearby target', numeric)
            require(abs(numeric['targetP'][0]-new_x)<1e-7, 'Numeric X field did not update target', numeric)
            np.testing.assert_array_equal(numeric['targetR'], numeric_start['targetR'],
                err_msg='Editing position rounded or changed the target orientation')
            np.testing.assert_array_equal(numeric['targetP'][1:], numeric_start['targetP'][1:],
                err_msg='Editing X changed untouched target coordinates')
            old_roll = await browser.evaluate("Number(document.getElementById('arm-roll').value)")
            await fill_number(browser, '#arm-roll', round(old_roll+2, 1))
            oriented = await browser.evaluate(STATE)
            require(oriented['matched'], 'Numeric orientation edit failed for a nearby target', oriented)
            np.testing.assert_array_equal(oriented['targetP'], numeric['targetP'],
                err_msg='Editing orientation rounded or changed the target position')
            require(np.max(np.abs(np.asarray(oriented['targetR'])-numeric['targetR']))>.01,
                    'Numeric roll field did not rotate the target', oriented)
            report['checks']['numeric_position_and_orientation'] = dict(position_error_m=oriented['positionError'],
                orientation_error_rad=oriented['orientationError'],
                position_edit_preserves_exact_orientation=True, orientation_edit_preserves_exact_position=True,
                position_edit_preserves_untouched_coordinates=True)

            # Preserve the last reached configuration when a target is impossible.
            before_invalid = await browser.evaluate(STATE)
            await fill_number(browser, '#arm-target-x', 5)
            invalid = await browser.evaluate(STATE)
            require(not invalid['matched'] and invalid['cap']=='—', 'Impossible target retained a payload rating', invalid)
            require('beyond' in invalid['status'].lower(), 'Unreachable reason not explained', invalid)
            np.testing.assert_allclose(invalid['q'], before_invalid['q'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(invalid['p'], before_invalid['p'], atol=1e-12, rtol=0)
            await click(browser, '#arm-reset')
            reset = await browser.evaluate(STATE)
            require(reset['matched'] and reset['cap']!='—', 'Reset did not restore a valid target/readout', reset)
            np.testing.assert_allclose(reset['q'], before_invalid['q'], atol=1e-12, rtol=0)
            report['checks']['unreachable_target_and_reset'] = dict(kept_last_pose=True, hidden_target_capacity=True,
                rejected_position_error_m=invalid['positionError'])

            await click(browser, '#workspace-mode')
            require((await browser.evaluate(STATE))['mode']=='workspace', 'Workspace mode did not open')
            selection = await browser.evaluate('''(() => {
              const i=workspace.current.capacity.findIndex(value=>value>=2);
              workspace.selectPose(i);
              return {index:i,q:Array.from(workspace.current.angles.subarray(i*6,i*6+6)),tcp:workspace.state.pose.tcp};
            })()''')
            require(selection['index']>=0, 'No suitable saved map pose available')
            await click(browser, '#arm-mode')
            require(not await browser.evaluate("document.getElementById('arm-adopt').disabled"), 'Map adoption button stayed disabled')
            await click(browser, '#arm-adopt')
            adopted = await browser.evaluate(STATE)
            np.testing.assert_allclose(adopted['q'], np.arctan2(np.sin(selection['q']),np.cos(selection['q'])), atol=1e-12)
            np.testing.assert_allclose(adopted['p'], selection['tcp'], atol=1e-10)
            require(adopted['matched'] and adopted['mode']=='arm', 'Adopted pose not active in arm editor', adopted)
            report['checks']['workspace_switch_and_pose_adoption'] = dict(index=selection['index'])

            # A rod preview must update this editor's model and mass/torque
            # calculation, then restore the original model without a stale pose.
            await prepare_pose(browser)
            original_model_pose = await browser.evaluate(STATE)
            changed = await browser.evaluate('''(async () => {
              await workspace.setDesignLengths([350,350]);armEditor.draw();
              return {sameModel:armEditor.model===workspace.getModel(),design:workspace.state.design,
                expected:ArmPoseSolver.forward(armEditor.state.q,workspace.getModel()).tcp};
            })()''')
            await frame(browser)
            changed_pose = await browser.evaluate(STATE)
            require(changed['sameModel'] and changed['design'], 'Editor retained the old rod model', changed)
            np.testing.assert_allclose(changed_pose['q'], original_model_pose['q'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(changed_pose['p'], changed['expected'], atol=1e-12, rtol=0)
            require(np.linalg.norm(np.asarray(changed_pose['p'])-original_model_pose['p'])>.05,
                    'Changed rods did not move the grasp point', changed_pose)
            require(changed_pose['cap']!=original_model_pose['cap'],
                    'Changed rods did not refresh the payload limit', changed_pose)
            restored = await browser.evaluate('''(() => {
              workspace.showOriginal();armEditor.draw();
              return {sameModel:armEditor.model===workspace.getModel(),design:workspace.state.design};
            })()''')
            await frame(browser)
            restored_pose = await browser.evaluate(STATE)
            require(restored['sameModel'] and not restored['design'], 'Original rod model did not restore', restored)
            np.testing.assert_allclose(restored_pose['p'], original_model_pose['p'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(restored_pose['q'], original_model_pose['q'], atol=1e-12, rtol=0)
            require(restored_pose['cap']==original_model_pose['cap'], 'Original payload limit did not restore', restored_pose)
            report['checks']['live_rod_model_change_and_restore'] = dict(lengths_mm=[350,350],
                capacity_before=original_model_pose['cap'],capacity_changed=changed_pose['cap'],
                capacity_restored=restored_pose['cap'],
                grasp_displacement_m=float(np.linalg.norm(np.asarray(changed_pose['p'])-original_model_pose['p'])))

            # Representative desktop image and overflow check.
            await browser.evaluate("armEditor.setMode('arm');armEditor.setQ([0,0,0,0,0,0]);armEditor.setTool('move');armEditor.setView('three')")
            await frame(browser)
            await browser.evaluate("document.querySelectorAll('details').forEach(d=>d.open=false);scrollTo(0,0)")
            await frame(browser)
            desktop = await browser.evaluate('({viewport:innerWidth,document:document.documentElement.scrollWidth,stage:document.getElementById("arm-stage").clientWidth})')
            require(desktop['document']<=desktop['viewport'], 'Desktop horizontal page overflow', desktop)
            desktop_path = output/'manipulator_preview.png'
            await browser.screenshot(desktop_path)
            report['screenshots'].append(str(desktop_path))
            report['checks']['desktop_layout'] = desktop

            await browser.command('Emulation.setDeviceMetricsOverride', width=390, height=900, deviceScaleFactor=1, mobile=True)
            await frame(browser)
            await visible_rect(browser, '#arm-target-x')
            await frame(browser)
            mobile = await browser.evaluate('''(() => {
              const fields=['arm-target-x','arm-target-y','arm-target-z','arm-roll','arm-pitch','arm-yaw'];
              return {viewport:innerWidth,document:document.documentElement.scrollWidth,
                stage:document.getElementById('arm-stage').clientWidth,
                fields_inside:fields.every(id=>{const r=document.getElementById(id).getBoundingClientRect();return r.width>0&&r.height>0&&r.left>=0&&r.right<=innerWidth}),
                precise_panel_open:document.getElementById('arm-target-x').closest('details').open};
            })()''')
            # Mobile emulation widens innerWidth to fit overflowing content, so compare with the 390 px screen.
            require(mobile['document']<=390 and mobile['viewport']<=390 and mobile['fields_inside'] and mobile['precise_panel_open'], 'Visible mobile controls overflow the page or remain collapsed', mobile)
            require(mobile['stage']>200, 'Mobile drawing area collapsed', mobile)
            await click(browser, '#open-geometry')
            await click(browser, '#design-map')
            await browser.wait_for('workspace.state.design && !workspace.design.worker')
            report['checks']['mobile_geometry_dialog'] = await check_dialog_fit(browser)
            await click(browser, '#done-geometry')
            await browser.evaluate("workspace.showOriginal();document.querySelectorAll('details').forEach(d=>d.open=false);scrollTo(0,0);armEditor.draw()")
            await frame(browser)
            mobile_path = output/'manipulator_mobile_preview.png'
            await browser.screenshot(mobile_path)
            report['screenshots'].append(str(mobile_path))
            report['checks']['mobile_layout'] = mobile
            report['browser_errors'] = browser.errors
            require(not browser.errors, 'Browser emitted JavaScript exceptions', browser.errors)
            report['passed'] = True
        except Exception as error:
            report['error'] = str(error)
            report['traceback'] = traceback.format_exc()
            report['browser_errors'] = browser.errors
            try:
                path = output/'manipulator_failure.png'
                await browser.screenshot(path)
                report['screenshots'].append(str(path))
                report['last_state'] = await browser.evaluate(STATE)
            except Exception:
                pass
        finally:
            (output/'manipulator_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = asyncio.run(run(args.output.resolve()))
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
